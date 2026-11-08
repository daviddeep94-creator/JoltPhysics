// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#include <Jolt/Physics/Collision/Shape/VoxelShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/CompoundShape.h>
#include <Jolt/Physics/Collision/Shape/PolyhedronSubmergedVolumeCalculator.h>
#include <Jolt/Physics/Collision/CollisionDispatch.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollidePointResult.h>
#include <Jolt/Physics/Collision/CollideSoftBodyVertexIterator.h>
#include <chrono>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <algorithm>
#include <functional>
#include <thread>
#include <Jolt/Physics/Collision/TransformedShape.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Core/Atomics.h>
#include <Jolt/Core/Profiler.h>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
	#include <intrin.h>		// __popcnt64（Recalculate 的字级统计用）
#endif
#ifdef JPH_DEBUG_RENDERER
	#include <Jolt/Renderer/DebugRenderer.h>
#endif // JPH_DEBUG_RENDERER

JPH_NAMESPACE_BEGIN

/// 全局材质表：按**方块编号**索引（0 = 空 = 用形状的默认材质）。
/// ★ 所有 VoxelShape 共用这一份表 ⇒ 每个形状零额外内存
///   （否则 2800 个地形块 × 256 项会是几十 MB）。
/// ★★ 这张表是**写时复制**的（见下面的 sVoxelMaterialTable）：
///   读侧（PhysicsSystem 的 CombineFriction / CombineRestitution 回调、CharacterVirtual 的支撑接触）
///   跑在多个 worker 线程上，而写侧（Unity 主线程切关卡 / 改材质配置）随时可能发生 —— 原地改这
///   256 项是货真价实的数据竞争（32 位 ARM 上还会读到撕裂的 float）。所以 sSetMaterialTable()
///   改成"新建一张表 -> 填 -> 原子发布"，读侧每次只多做一次 atomic load（x86 上就是一条 mov）。
/// ★ 旧表**永不释放**：GetMaterial() 的返回值会被上层塞进 RefConst，可能有线程还握着它的指针。
///   一张表 256 项 × 24 字节 ≈ 6 KB，换表频率是"进关卡"级别 ⇒ 这点"泄漏"是有意为之。
static VoxelMaterial sVoxelMaterials[256];

//////////////////////////////////////////////////////////////////////////////////////////
// 撞击破坏：物理步进期间的「逐格材质强度」累加 —— **按形状对记账**
//
// 为什么按 (shape1, shape2) 而不是 thread_local：OnContactAdded 与产生接触的 CollideShape
// 在 Jolt 的 job 图里是**不同阶段** —— 同一 worker 线程在两者之间会处理**别的 body pair**，
// thread_local 早就被覆盖（实测：记账 30 万次、回调仍读 0）。按形状对存，回调时拿
// 两个 body 的形状指针就能查到属于这一对的记录。
//
// 线程模型：每个 body pair 每步只被一个线程处理 ⇒ 记录的查重/占位走互斥锁，
// 之后的格号/强度写入是单写者、不加锁；读侧（回调线程 / 步进后的主线程）与
// 其它 pair 的写入互不干扰，读快照时走同一把短临界区锁。
// 生命周期：每步开始 sBeginStepPairs() 清空（上层在物理步进前调用）。
//////////////////////////////////////////////////////////////////////////////////////////
// ★★ 形状对表上限：1024（2026-11-06）→ 4096（2026-11-07）。
//   表满 ⇒ sStepPairClaim 返回 nullptr ⇒ 这对**整片不记账** ⇒ 判据读 0 ⇒ 不挖洞，
//   球继续往里钻、下步重叠更大、形状对更多、丢得更多 ⇒ 穿透深度单调递增（Unity 现场
//   「记录 7764→12335 格、穿透 0.036→0.09 m」）。扩表前每条按逐格上界 ~6k 格 ×2 侧 ×4B
//   = 48KB，4096 条 = 191MB 不可接受；4³ 块记账后每条上界按**块数**算（16³ 叶子对 ≈ 343 块，
//   见 sStepPairClaim），4096 条 ≈ 11MB ⇒ 可以放开。
static constexpr uint sMaxStepPairRecords = 4096;	// 一步内最多多少个形状对

/// ★★★ B 方案「块抽样」的门槛（2026-11-06）：遍历侧重叠区间（range）的格数 ≥ 该值时，
///   本次查询的接触**和记账**都按 2x2x2 块抽样（见 sCollideVoxelGrids 里 block_sample 的说明）。
///   小重叠（静止方块 ~100 格、角色脚下查询）保持原来的逐格路径**逐位不变**；
///   大重叠（深嵌球、碎块撞击，每对几千~几万格）才切块路径 —— 那里逐格路径一帧能走 6 万格，
///   是帧率崩掉的根源。1024 = 2¹⁰（16³ 叶子对窗口 ≈ 半个球冠，像样的重叠都会过线；
///   再抬到 4096 会让复合体叶子走逐格、单网格走块路径 ⇒「复合 ≡ 单网格」等价直接爆炸）。
long long VoxelShape::sBlockSampleMinVolume = 1024;

/// ★ 记账块级化（默认开；关掉 = 回到旧的逐格记账，仅供单测 A/B）：
///   开启后 run_cell 的候选侧不再逐格枚举，改按 4x4x4 块推进，每块一次 GetWord + 一次球心距，
///   记 1 条**块号**（bit31 打标，C# 侧按整 4³ 字清零）。接触几何完全不受影响。见 run_cell。
bool VoxelShape::sRecordBlockAgg = true;
int  VoxelShape::sRecordBlockShift = 2;

/// 每对每侧的格号记录容量（运行时可调，见 VoxelShapeDebug.MaxCellsPerSide）。
/// 超出截断：少挖几格，不越界。默认 8192 已经覆盖「5m 球 16 m/s 一步 4.4 格」的球冠（约 2.3k 格）。
int VoxelShape::sMaxCellsPerSide = 8192;

struct StepPairRecord
{
	const void *	mShape1;						// 这次碰撞调用的 shape1（体素或非体素）
	const void *	mShape2;
	float			mStrength1;						// shape1 侧承压和（非体素侧恒 0）
	float			mStrength2;
	uint			mCount1;						// shape1 侧格号数
	uint			mCount2;
	uint			mDedupGen;						// ★ 本条记录的"代数"：格去重哈希表靠它区分是谁的槽位（0 = 未分配）
	// ★★ 格号缓冲在 claim 时一次性分配（不是定长数组、也不是会自增长的 vector）：
	//   高速深嵌时单侧相交格可达数万，4096 定长会把「这一击撞到的格」截断 ⇒ 洞挖不干净。
	//   ★ 不能用 std::vector：记录是**按形状对共享**的，多线程并发查询同一对时会同时 push_back
	//     ⇒ 扩容时的 reallocate 直接踹掉别的线程正在写的缓冲（单测的多线程扩展性用例实测崩溃）。
	//     改成"claim 时按自适应上界分配一次、之后只写不扩"，容量变化只在 claim（锁内）时发生。
	//     上界 = 两侧都是体素时 (较小方三轴+2) 的乘积（见 sStepPairClaim），否则 = 全局 MaxCellsPerSide。
	uint32_t *		mCells1 = nullptr;
	uint32_t *		mCells2 = nullptr;
	uint			mCap = 0;						// 上面两块缓冲的容量（格数）
};

/// 追加一颗格号到记录（超过 sMaxCellsPerSide 就丢：上层还有扩散补洞形）。
static inline void sRecordCell(StepPairRecord *inRecord, bool inSide1, uint32_t inCell)
{
	uint32_t *cells = inSide1? inRecord->mCells1 : inRecord->mCells2;
	uint &count = inSide1? inRecord->mCount1 : inRecord->mCount2;
	if (cells == nullptr || count >= inRecord->mCap)
		return;
	cells[count++] = inCell;
}

static mutex			sStepPairMutex;
static StepPairRecord	sStepPairRecords[sMaxStepPairRecords];
static uint				sStepPairRecordCount = 0;

//////////////////////////////////////////////////////////////////////////////////////////
// ★ 整对 body 跨子 shape 的 Σ承压累加器（2026-11-09 修正「逐子 shape 判破 ⇒ 穿透不削体素」）。
//   一个 body 可能挂多个 VoxelShape 子 shape；窄相位按 leaf pair 逐对派发，单对的 mStrength
//   只是该对的局部量。判破必须看「整对 body 跨所有子 shape 的重叠强度之和」⇒ 用 bodyID 无序对作 key 累加。
struct VoxelStepBodyPairAccum { float sSide1 = 0; float sSide2 = 0; bool broke = false; };
static std::unordered_map<uint64, VoxelStepBodyPairAccum> sThisStepStrength[1024];
static inline uint64 sBodyPairKey(uint32 a, uint32 b) { return a < b ? (uint64(a) << 32) | b : (uint64(b) << 32) | a; }
// 跨线程并发写不同 bpk ⇒ 用分片互斥：每个 shard 是独立 map，按 bpk 取模选 shard，避免全局锁序列化记账。
static constexpr uint sStrengthShards = 1024;
static mutex sStrengthMutex[1024];

// ★★★ 提前判破（2026-11-08）：破帧不必再逐格算法线 / 接触点，也不必进求解器。
//
// 顺序（用户定）：① 全局轴 → ② 用全局轴投影估冲量 → ③ 4³ 记账 → ④ 判破 →
//                ⑤ 破则 return（不产出接触流形 ⇒ 求解器自然不参与）；不破则照常窄相位 + 求解。
//
// ①②③ 都拿到 Shape* 但拿不到 Body：Jolt 的 CollisionDispatch 只有形状 + 变换。
// 正解是步首由 joltc 从 PhysicsSystem 的 BodyManager 把每个 VoxelShape 叶子对应的
// Body 运动学量拷成快照（VoxelStepBody），窄相位按叶子指针 O(1) 查 —— 不改 dispatch 签名。
//
// 破帧没有流形 ⇒ C# 的 OnContactAdded 不会触发 ⇒ 挖洞 / 碎块 / 反向冲量链路会整条断掉。
// 所以这里把破结论记进 sStepBreakRecords，C# 在 Flush 开头读回转成 PairEvent。
//////////////////////////////////////////////////////////////////////////////////////////
static constexpr uint		sMaxStepBreakRecords = 1024;
static std::vector<VoxelStepBody>	sStepBodies;
static VoxelStepBreakConfig	sStepBreakCfg;
static bool					sStepBreakEnabled = false;
static std::vector<VoxelStepBreakRecord>	sStepBreakRecords;
static mutex				sStepBreakMutex;

// 叶子指针 → sStepBodies 下标的开放寻址表（稳态零分配；步首一次性建好）
static std::vector<const void *>	sStepBodyKeys;
static std::vector<uint>			sStepBodyVals;
static uint32				sStepBodyMask = 0;

static inline uint64 sStepBodyHash(const void *inLeaf)
{
	uint64 h = uint64(reinterpret_cast<uintptr_t>(inLeaf));
	h ^= h >> 33;
	h *= 0xff51afd7ed558ccdULL;
	h ^= h >> 33;
	return h;
}

static inline const VoxelStepBody *sStepBodyLookup(const void *inLeaf)
{
	if (sStepBodyMask == 0 || inLeaf == nullptr)
		return nullptr;
	uint32 i = uint32(sStepBodyHash(inLeaf)) & sStepBodyMask;
	for (;;)
	{
		const void *key = sStepBodyKeys[i];
		if (key == nullptr)
			return nullptr;
		if (key == inLeaf)
			return &sStepBodies[sStepBodyVals[i]];
		i = (i + 1) & sStepBodyMask;
	}
}

/// 这一侧是否可判破（对齐 ImpactListener::Handle 的 aBreakable / bBreakable）。
static inline bool sStepBreakable(const VoxelStepBody *b)
{
	if (b == nullptr || b->mKind == 0) return false;
	if (b->mKind == 3 && !sStepBreakCfg.mAllowDebrisToBreak) return false;
	return true;
}

/// ★ 第 2 步：用 1c) 算出的全局轴对速度向量做投影估冲量（在记账之前）。
///   J = (1+e)·|v_rel·axis| / (1/m1 + 1/m2)，与 Handle 的解析解同式，只把逐格
///   流形法线换成全局轴。不直接用重量 × 速度：斜撞时角度不同、碰撞力也不同。
///   返回 < 0 = 这对不具备提前判破资格（快照缺失 / 全静态 / 速度闸 / 排队门槛）。
static float sEstimateEarlyImpulse(const void *inLeaf1, const void *inLeaf2, Vec3Arg inAxis)
{
	const VoxelStepBody *b1 = sStepBodyLookup(inLeaf1);
	const VoxelStepBody *b2 = sStepBodyLookup(inLeaf2);
	if (b1 == nullptr || b2 == nullptr || b1->mBody == b2->mBody)
		return -1.0f;

	const bool can1 = sStepBreakable(b1), can2 = sStepBreakable(b2);
	if (!can1 && !can2) return -1.0f;
	if (!b1->mIsDynamic && !b2->mIsDynamic) return -1.0f;
	// Handle 的「静态侧不是任何可破坏体素」早退
	if ((!can1 && !b1->mIsDynamic) || (!can2 && !b2->mIsDynamic)) return -1.0f;

	const Vec3 v1(b1->mVel[0], b1->mVel[1], b1->mVel[2]);
	const Vec3 v2(b2->mVel[0], b2->mVel[1], b2->mVel[2]);
	const Vec3 vrel = v1 - v2;
	if (vrel.Length() < sStepBreakCfg.mMinImpactSpeed * 0.2f) return -1.0f;
	const float invSum = b1->mInvMass + b2->mInvMass;
	if (invSum <= 0.0f) return -1.0f;

	const float e = std::max(0.0f, std::max(b1->mRestitution, b2->mRestitution));
	const float vn = vrel.Dot(inAxis);
	const float J = (1.0f + e) * (vn >= 0.0f? vn : -vn) / invSum;
	return J < sStepBreakCfg.mQueueFloor? -1.0f : J;
}

/// ★ 第 4 步（记账之后）：拿估冲量对比本对叶子的 Σ承压，破则记一条破记录。
/// 返回 true = 本对叶子的接触几何可以整个跳过（⑤ 不再执行）。
static bool sTryEarlyBreak(const void *inLeaf1, const void *inLeaf2,
                           uint64 inBpk, float inImpulse,
                           Vec3Arg inAxis, const AABox &inMeetWorld)
{
	if (inImpulse < 0.0f)
		return false;
	const VoxelStepBody *b1 = sStepBodyLookup(inLeaf1);
	const VoxelStepBody *b2 = sStepBodyLookup(inLeaf2);
	if (b1 == nullptr || b2 == nullptr)
		return false;

	const bool can1 = sStepBreakable(b1), can2 = sStepBreakable(b2);
	// ★ 整对 body 的全量 Σ承压 = 本步跨所有子 shape 累加的该对强度之和（inBpk 索引）。
	//   不再用单 leaf pair 的局部 mStrength，那正是「逐子 shape 判破 ⇒ 穿透但不削体素」的根因。
	float tot1 = 0.0f, tot2 = 0.0f;
	if (inBpk != 0)
	{
		const uint sh = uint(inBpk & (sStrengthShards - 1));
		lock_guard<mutex> lk(sStrengthMutex[sh]);
		auto it = sThisStepStrength[sh].find(inBpk);
		if (it != sThisStepStrength[sh].end()) { tot1 = it->second.sSide1; tot2 = it->second.sSide2; }
	}
	const bool has1 = tot1 > 0.0f;
	const bool has2 = tot2 > 0.0f;
	if (!has1 && !has2) return false;
	const bool br1 = can1 && has1 && (!sStepBreakCfg.mStrengthGate || inImpulse > tot1);
	const bool br2 = can2 && has2 && (!sStepBreakCfg.mStrengthGate || inImpulse > tot2);
	if (!br1 && !br2) return false;

	// ★ 标记本 body 对已判破（本步诊断用；判破结论由本函数返回 true 驱动，C# 据此挖洞）。
	if (inBpk != 0)
	{
		const uint sh = uint(inBpk & (sStrengthShards - 1));
		lock_guard<mutex> lk(sStrengthMutex[sh]);
		sThisStepStrength[sh][inBpk].broke = true;
	}

	const Vec3 v1(b1->mVel[0], b1->mVel[1], b1->mVel[2]);
	const Vec3 v2(b2->mVel[0], b2->mVel[1], b2->mVel[2]);
	const Vec3 point = inMeetWorld.IsValid()? (inMeetWorld.mMin + inMeetWorld.mMax) * 0.5f : Vec3::sZero();

	VoxelStepBreakRecord rec;
	rec.mBody1 = b1->mBody;
	rec.mBody2 = b2->mBody;
	rec.mBodyID1 = b1->mBodyID;
	rec.mBodyID2 = b2->mBodyID;
	rec.mImpulse = inImpulse;
	rec.mNormal[0] = inAxis.GetX(); rec.mNormal[1] = inAxis.GetY(); rec.mNormal[2] = inAxis.GetZ();
	rec.mPoint[0] = point.GetX();  rec.mPoint[1] = point.GetY();  rec.mPoint[2] = point.GetZ();
	rec.mVel1[0] = v1.GetX(); rec.mVel1[1] = v1.GetY(); rec.mVel1[2] = v1.GetZ();
	rec.mVel2[0] = v2.GetX(); rec.mVel2[1] = v2.GetY(); rec.mVel2[2] = v2.GetZ();
	rec.mInvMass1 = b1->mInvMass;
	rec.mInvMass2 = b2->mInvMass;
	rec.mBreak1 = br1? 1 : 0;
	rec.mBreak2 = br2? 1 : 0;
	rec.mKind1 = b1->mKind;
	rec.mKind2 = b2->mKind;

	lock_guard<mutex> lock(sStepBreakMutex);
	// 同一个 body 对会有多个叶子对：合成一条（冲量累加，与 Handle 多流形同口径），
	// 免得 C# Flush 里 PairKey 去重把后面几条丢掉。
	for (uint i = 0; i < sStepBreakRecords.size(); ++i)
	{
		VoxelStepBreakRecord &prev = sStepBreakRecords[i];
		const bool same = (prev.mBodyID1 == rec.mBodyID1 && prev.mBodyID2 == rec.mBodyID2);
		const bool flip = (prev.mBodyID1 == rec.mBodyID2 && prev.mBodyID2 == rec.mBodyID1);
		if (!same && !flip) continue;
		prev.mImpulse += rec.mImpulse;
		// 反向记进来的叶子对（body1/body2 与首条相反）⇒ 破标志也要跟着换边，
		// 否则合并后 break1/break2 会落到错的 body 上。
		if (same)
		{
			prev.mBreak1 |= rec.mBreak1;
			prev.mBreak2 |= rec.mBreak2;
		}
		else
		{
			prev.mBreak1 |= rec.mBreak2;
			prev.mBreak2 |= rec.mBreak1;
		}
		return true;
	}
	if (sStepBreakRecords.size() < sMaxStepBreakRecords)
		sStepBreakRecords.push_back(rec);
	return true;
}

void VoxelShape::sSetStepBodies(const VoxelStepBody *inBodies, uint inCount,
                                const VoxelStepBreakConfig &inConfig)
{
	sStepBodies.clear();
	if (inBodies != nullptr && inCount > 0)
		sStepBodies.assign(inBodies, inBodies + inCount);
	sStepBreakCfg = inConfig;

	uint32 slots = 64;
	while (slots < uint32(inCount) * 2u && slots < (1u << 20))
		slots <<= 1;
	sStepBodyKeys.assign(slots, nullptr);
	sStepBodyVals.assign(slots, 0);
	sStepBodyMask = slots - 1;
	for (uint i = 0; i < sStepBodies.size(); ++i)
	{
		const void *leaf = sStepBodies[i].mLeaf;
		if (leaf == nullptr) continue;
		uint32 k = uint32(sStepBodyHash(leaf)) & sStepBodyMask;
		while (sStepBodyKeys[k] != nullptr)
			k = (k + 1) & sStepBodyMask;
		sStepBodyKeys[k] = leaf;
		sStepBodyVals[k] = i;
	}
	sStepBreakEnabled = inConfig.mEnabled != 0
	                 && inConfig.mForceFullNarrow == 0
	                 && !sStepBodies.empty();
	lock_guard<mutex> lock(sStepBreakMutex);
	sStepBreakRecords.clear();
}

void VoxelShape::sClearStepBodies()
{
	sStepBodies.clear();
	sStepBodyKeys.clear();
	sStepBodyVals.clear();
	sStepBodyMask = 0;
	sStepBreakEnabled = false;
	lock_guard<mutex> lock(sStepBreakMutex);
	sStepBreakRecords.clear();
}

uint VoxelShape::sGetStepBreakRecords(VoxelStepBreakRecord *outRecords, uint inCapacity)
{
	lock_guard<mutex> lock(sStepBreakMutex);
	if (outRecords == nullptr || inCapacity == 0)
		return uint(sStepBreakRecords.size());
	const uint n = std::min<uint>(uint(sStepBreakRecords.size()), inCapacity);
	memcpy(outRecords, sStepBreakRecords.data(), size_t(n) * sizeof(VoxelStepBreakRecord));
	return n;
}

/// 把一格编成 x | y<<10 | z<<20（各 10 位）。越界轴直接钳住：宁可坐标错、不可越位污染其它轴。
static inline uint32_t sPackCell(uint inX, uint inY, uint inZ)
{
	return (inX & 0x3FFu) | ((inY & 0x3FFu) << 10) | ((inZ & 0x3FFu) << 20);
}

//////////////////////////////////////////////////////////////////////////////////////////
// 承压按「格」去重
//
// 为什么需要：run_cell 是「遍历侧一颗格 × 候选侧跨格窗口（每轴 |Δ| < 1）」，相邻遍历格的
// 窗口互相重叠 ⇒ **同一颗候选格**会和周围 2~4 颗遍历格各配成一个格对。格对本身不重复，
// 但承压是**材质属性、属于格**：一颗格不管被几颗对方格顶到，强度只有一份。按格对累加会让
// Σ承压 虚高 2~4 倍，而且倍数随穿透几何漂移（同样的撞击时灵时不灵）。
//
// 两侧的重复性质不同：
//   · 遍历侧格号（inX/inY/inZ）在一颗 run_cell 内是**常量** ⇒ 一个 bool 就够，零查表；
//   · 候选侧格号（x/y/z）会被相邻遍历格重复枚举 ⇒ 必须查表。
//
// 为什么用哈希不用线性扫 mCells：格数上限 1024，线性扫每对平均 n/2 次比较（一次撞击 ~50 万次），
// 哈希 1~2 次探测。
//
// 为什么零清表：槽里存 {格号, 代数}，代数 != 当前 record 的代数即视为空槽。换一条 record
// 只是代数不同，不需要 memset（256 条记录 × 4096 槽逐条清是不可接受的每步开销）。
// thread_local：一个线程同一时刻只处理一条记录；串到别的记录再回来时旧槽可能已被覆盖
// ⇒ 退化成"少去重一点"（多算承压 = 更难破坏，安全方向），不会误判成"已记过"而漏格。
//////////////////////////////////////////////////////////////////////////////////////////
// 槽数 = 4 ×「单侧格数上限」（负载 25%），随 MaxContactPoints 自动放大，见 sDedupFirstHit。
// ★ 之前写死 4096 槽（只对应 1024 格）：把 MaxContactPoints 调到 4096 之后负载变 100%，
//   线性探测退化成整表扫描，且表满后本函数一律返回 true ⇒ 同一颗格被重复记录，
//   「记录 4096 格」里大量是重复格 —— 这是"调到 4096 还是不够"的直接原因之一。
static thread_local std::vector<uint32_t> sDedupKey;
static thread_local std::vector<uint>     sDedupGenStore;	// 0 = 从未用过（代数从 1 开始分配）
static thread_local uint sDedupSlotCount = 0;				// 0 = 本线程尚未分配
static std::atomic<uint> sDedupWantedSlots(4096);			// 主线程每步开头算好（2 的幂）
static uint sDedupGenCounter = 0;								// 只在 sStepPairClaim 的锁内自增

/// 这颗格在本条记录里是不是**第一次**出现（true ⇒ 该累加强度、该记格号）。
static inline bool sDedupFirstHit(uint inGen, uint32_t inCell)
{
	const uint want = sDedupWantedSlots.load(std::memory_order_relaxed);
	if (sDedupSlotCount != want)
	{
		// 容量变了（或本线程首次使用）⇒ 重建：全 0 = 全部空槽。
		// 重建只发生在步首/首次调用，此时本线程本步还没有有效槽位 ⇒ 不会误判"已记过"，只会多记。
		sDedupKey.assign(want, 0);
		sDedupGenStore.assign(want, 0);
		sDedupSlotCount = want;
	}
	const uint slots = sDedupSlotCount;
	uint i = (inCell * 2654435761u) & (slots - 1);
	for (uint n = 0; n < slots; ++n)
	{
		if (sDedupGenStore[i] != inGen)			// 空槽（属于别的记录 / 从没用过）
		{
			sDedupGenStore[i] = inGen;
			sDedupKey[i] = inCell;
			return true;
		}
		if (sDedupKey[i] == inCell)
			return false;
		i = (i + 1) & (slots - 1);
	}
	return true;							// 表满（槽数 = 4×格数上限，不该发生）⇒ 按首次处理，宁可多算
}

/// 查重 / 占位一条形状对记录（调用方 = 窄相位入口，拿到的指针之后单写者使用）。
/// 同一对重复进来（理论上一步只跑一次，查询路径除外）⇒ 继续累加同一份。
/// 表满 ⇒ 返回 nullptr：这一对不记账（读侧拿到 0 ⇒ 判不了破坏 ⇒ 正常求解，安全兜底）。
static StepPairRecord *sStepPairClaim(const void *inShape1, const void *inShape2)
{
	lock_guard<mutex> lock(sStepPairMutex);
	for (uint i = 0; i < sStepPairRecordCount; ++i)
	{
		StepPairRecord &r = sStepPairRecords[i];
		if ((r.mShape1 == inShape1 && r.mShape2 == inShape2) ||
			(r.mShape1 == inShape2 && r.mShape2 == inShape1))
			return &r;
	}
	if (sStepPairRecordCount >= sMaxStepPairRecords)
		return nullptr;
	StepPairRecord &r = sStepPairRecords[sStepPairRecordCount++];
	r.mShape1 = inShape1;
	r.mShape2 = inShape2;
	r.mStrength1 = r.mStrength2 = 0.0f;
	r.mCount1 = r.mCount2 = 0;
	// 格号缓冲：容量自适应 —— 只在容量变了时重分配（这里在锁内 ⇒ 独占）；记录跨步复用 ⇒ 稳态零分配。
	// ★ 两侧都是体素形状时，重叠窗口被**较小一方**的格盒框死（遍历 = 窗口内的实心格 × 窗口 ±1 格的
	//   候选格）⇒ 每侧格数上界 = (较小方三轴各 +2) 的乘积。按这个上界分配，别按全局 MaxCellsPerSide：
	//   16³ 叶子对每条 ~6k 格（~48KB），按全局上限（26.2 万格）一条 = 2MB —— 大 compound 一步几百条
	//   记录就是几百 MB ⇒ 表只能做小 ⇒ 表溢出整片丢记账（"洞缺一整块"的根源）。
	//   单大网格 × 单大网格（分块失败回退 / 体素体对撞）没有叶子上界 ⇒ 仍按全局上限。
	{
		uint want_cap = uint(VoxelShape::sMaxCellsPerSide);
		const VoxelShape *vs1 = static_cast<const Shape *>(inShape1)->GetSubType() == EShapeSubType::User1? static_cast<const VoxelShape *>(inShape1) : nullptr;
		const VoxelShape *vs2 = static_cast<const Shape *>(inShape2)->GetSubType() == EShapeSubType::User1? static_cast<const VoxelShape *>(inShape2) : nullptr;
		if (vs1 != nullptr && vs2 != nullptr)
		{
			// 重叠窗口 ⊆ 较小一方的**实心格包围范围**（GetSolidMin/Max：公开、未统计时回退整格、不触发扫描）。
			// 每侧格数上界 = (实心范围各轴 + 2) 的乘积（候选侧经 straddle 最多再探出 1 格）。
			const VoxelShape &small = [&]() -> const VoxelShape &
			{
				uint64 n1 = 1, n2 = 1;
				for (int k = 0; k < 3; ++k)
				{
					n1 *= uint64(vs1->GetSolidMax(k) - vs1->GetSolidMin(k) + 1);
					n2 *= uint64(vs2->GetSolidMax(k) - vs2->GetSolidMin(k) + 1);
				}
				return n1 <= n2? *vs1 : *vs2;
			}();
			uint64 bound = 1;
			for (int k = 0; k < 3; ++k)
				bound *= uint64(small.GetSolidMax(k) - small.GetSolidMin(k) + 3);
			if (VoxelShape::sRecordBlockAgg)
			{
				// ★ 4³ 块记账：一条记录 = 一个块角，不是一颗格。容量按**块数**给，
				//   否则 4096 条 × 6k 格 × 2 侧 × 4B = 191MB。每轴块数 =
				//   ceil((实心跨度 + 3) / 4) + 2 块窗口余量；16³ 叶子 ≈ 8³ = 512 块/侧
				//   （实际最多 4³ = 64 块）⇒ 一条 4KB，整表 16MB。
				//   关掉 sRecordBlockAgg 回到按格算，A/B 基线不受影响。
				uint64 blocks = 1;
				for (int k = 0; k < 3; ++k)
					blocks *= uint64((small.GetSolidMax(k) - small.GetSolidMin(k) + 6) / 4 + 2);
				bound = blocks;
			}
			if (bound < uint64(want_cap))
				want_cap = uint(bound);
		}
		if (r.mCap != want_cap)
		{
			delete [] r.mCells1;
			delete [] r.mCells2;
			r.mCells1 = want_cap > 0? new uint32_t[want_cap] : nullptr;
			r.mCells2 = want_cap > 0? new uint32_t[want_cap] : nullptr;
			r.mCap = want_cap;
		}
	}
	// 新记录换一个代数 ⇒ 格去重哈希表里属于上一条记录的槽位自动视为空槽（零清表）
	if (++sDedupGenCounter == 0)
		sDedupGenCounter = 1;				// 0 被用作"从未用过"
	r.mDedupGen = sDedupGenCounter;
	return &r;
}

void VoxelShape::sBeginStepPairs()
{
	// 格去重表容量随「单侧格数上限」走（负载 25%），向上取 2 的幂、最小 4096。
	// 工作线程在 sDedupFirstHit 里发现容量变了就重建自己那份（thread_local）。
	uint want = uint(std::max(VoxelShape::sMaxCellsPerSide, 1024)) * 4;
	uint slots = 4096;
	while (slots < want && slots < (1u << 21))
		slots <<= 1;
	sDedupWantedSlots.store(slots, std::memory_order_relaxed);

	lock_guard<mutex> lock(sStepPairMutex);
	sStepPairRecordCount = 0;

	// ★ 每步清空「整对 body 跨子 shape 累加器」，避免上一帧的强度残留在这一帧误判。
	for (uint i = 0; i < sStrengthShards; ++i)
		sThisStepStrength[i].clear();
}

uint VoxelShape::sGetStepPairData(const void *inVoxelShape, const void *inOtherShape,
                                  float *outStrength, uint32_t *outCells, uint inCapacity)
{
	lock_guard<mutex> lock(sStepPairMutex);
	for (uint i = 0; i < sStepPairRecordCount; ++i)
	{
		const StepPairRecord &r = sStepPairRecords[i];
		bool side1 = r.mShape1 == inVoxelShape && r.mShape2 == inOtherShape;
		bool side2 = r.mShape1 == inOtherShape && r.mShape2 == inVoxelShape;
		if (!side1 && !side2)
			continue;
		if (outStrength != nullptr)
			*outStrength = side1? r.mStrength1 : r.mStrength2;
		uint count = side1? r.mCount1 : r.mCount2;
		if (outCells == nullptr || inCapacity == 0)
			return count;
		if (count > inCapacity)
			count = inCapacity;					// 只拷能放下的部分：上层宁可少挖几格，也不要越界
		const uint32_t *src = side1? r.mCells1 : r.mCells2;
		if (count > 0 && src != nullptr && count <= r.mCap)
			memcpy(outCells, src, count * sizeof(uint32_t));
		else
			count = 0;
		return count;
	}
	if (outStrength != nullptr)
		*outStrength = 0.0f;
	return 0;
}

/// 递归收集根形状下所有叶子 VoxelShape（分块体 = compound 挂多个 VoxelShape 子形状；
/// 非复合时根自己就是叶子）。其它子类型直接忽略 —— 它们不参与体素记账。
static void sCollectVoxelLeaves(const Shape *inShape, Array<const void *> &ioLeaves)
{
	if (inShape == nullptr)
		return;
	if (inShape->GetSubType() == EShapeSubType::User1)
	{
		ioLeaves.push_back(inShape);
		return;
	}
	if (inShape->GetSubType() == EShapeSubType::StaticCompound
	 || inShape->GetSubType() == EShapeSubType::MutableCompound)
	{
		const CompoundShape *compound = static_cast<const CompoundShape *>(inShape);
		for (uint i = 0; i < compound->GetNumSubShapes(); ++i)
			sCollectVoxelLeaves(compound->GetSubShape(i).mShape.GetPtr(), ioLeaves);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////
// 读侧的「叶子指针 → 两侧下标」索引
//
// 旧实现：每条 StepPairRecord 用 std::find 线性扫 wallLeaves / punchLeaves 各一次，
//   再用 sLeafIndex 线性扫一遍拿下标。分块体 4096 叶 × 一步几千条记录 = 千万次
//   指针比较，全落在 Jolt/Step 的 C# 回调读侧。
// 新实现：把两张叶子表哈希成一张「指针 → (墙下标+1) | (撞击下标+1)<<16」的表，
//   每条记录只做两次 O(1) 探测。语义与 std::find / sLeafIndex 完全一致：
//   · 同一指针可以同时落在两侧 ⇒ 两个半字各自保留，side1 / side2 都能判；
//   · 查不到返回 0（有 side 判定时必查得到，减 1 即 0-based 下标）。
// ★ 代数标记（同 sDedupFirstHit）：每次建表 ++gen，旧槽自动失效 ⇒ 零清表；
//   thread_local + 稳态零分配，读侧在 sStepPairMutex 临界区内。
//////////////////////////////////////////////////////////////////////////////////////////
struct sLeafIndexSlot
{
	const void *	mKey = nullptr;
	uint			mGen = 0;
	uint32_t		mValue = 0;			// 低 16 位 = 墙侧下标+1，高 16 位 = 撞击侧下标+1
};
static thread_local std::vector<sLeafIndexSlot>	sLeafIndexSlots;
static thread_local uint						sLeafIndexGen = 0;

static inline uint32_t sLeafPtrHash(const void *inPtr)
{
	// shape 对象一般 16/32 字节对齐 ⇒ 低位恒 0，先右移再乘黄金比例打散
	uint64 v = uint64(reinterpret_cast<size_t>(inPtr)) >> 4;
	return uint32(v * 2654435761ull);
}

static void sLeafIndexBuild(const Array<const void *> &inWall, const Array<const void *> &inPunch)
{
	uint want = 16;
	const uint need = uint(inWall.size() + inPunch.size()) * 2 + 1;
	while (want < need)
		want <<= 1;
	if (uint(sLeafIndexSlots.size()) != want)
		sLeafIndexSlots.assign(want, sLeafIndexSlot());
	if (++sLeafIndexGen == 0)
	{
		for (sLeafIndexSlot &s : sLeafIndexSlots)
			s.mGen = 0;
		sLeafIndexGen = 1;
	}

	const uint mask = want - 1;
	auto add = [mask](const void *inKey, uint inIdx, uint inShift)
	{
		uint i = sLeafPtrHash(inKey) & mask;
		while (sLeafIndexSlots[i].mGen == sLeafIndexGen && sLeafIndexSlots[i].mKey != inKey)
			i = (i + 1) & mask;
		if (sLeafIndexSlots[i].mGen != sLeafIndexGen)
		{
			sLeafIndexSlots[i].mGen = sLeafIndexGen;
			sLeafIndexSlots[i].mKey = inKey;
			sLeafIndexSlots[i].mValue = 0;
		}
		sLeafIndexSlots[i].mValue |= (inIdx + 1u) << inShift;
	};
	for (uint i = 0; i < inWall.size(); ++i)
		add(inWall[i], i, 0);
	for (uint i = 0; i < inPunch.size(); ++i)
		add(inPunch[i], i, 16);
}

static inline uint32_t sLeafIndexLookup(const void *inKey)
{
	const uint mask = uint(sLeafIndexSlots.size()) - 1;
	uint i = sLeafPtrHash(inKey) & mask;
	while (sLeafIndexSlots[i].mGen == sLeafIndexGen)
	{
		if (sLeafIndexSlots[i].mKey == inKey)
			return sLeafIndexSlots[i].mValue;
		i = (i + 1) & mask;
	}
	return 0;
}

uint VoxelShape::sGetStepPairDataAggregated(const void *inWallRoot, const void *inPunchRoot,
                                            float *outStrength, uint32_t *outCells, uint inCapacity, uint16_t *outLeafIdx,
                                            float *outStrength2, uint32_t *outCells2, uint inCapacity2, uint16_t *outLeafIdx2,
                                            uint *outCount2)
{
	if (outStrength != nullptr)
		*outStrength = 0.0f;
	if (outStrength2 != nullptr)
		*outStrength2 = 0.0f;
	if (outCount2 != nullptr)
		*outCount2 = 0;
	if (inWallRoot == nullptr || inPunchRoot == nullptr)
		return 0;

	// 两侧各收集叶子 VoxelShape（单 shape 时就是根自己）
	Array<const void *> wallLeaves, punchLeaves;
	sCollectVoxelLeaves(reinterpret_cast<const Shape *>(inWallRoot), wallLeaves);
	sCollectVoxelLeaves(reinterpret_cast<const Shape *>(inPunchRoot), punchLeaves);
	if (wallLeaves.empty() || punchLeaves.empty())
		return 0;

	// ★ 建一次叶子指针索引（O(叶数)），之后每条记录 O(1)；旧版是每条记录
	//   std::find 两张表 + 两次线性 sLeafIndex（4096 叶 × 几千条 = 千万次比较/步）。
	//   下标语义不变：叶子在收集序列里的序号（= C# 分块建体时 children 顺序）。
	// ★★ uint16_t 是硬要求：256³ 区块按 16³ 切 = 4096 个子形状，以前用 uint8_t
	//   截断成 i & 255 ⇒ C# 按 CShapeRanges[i & 255] 取原点 ⇒ 破洞偏移好几米。
	sLeafIndexBuild(wallLeaves, punchLeaves);

	lock_guard<mutex> lock(sStepPairMutex);
	float strength = 0.0f, strength2 = 0.0f;
	uint copied = 0, copied2 = 0;
	for (uint i = 0; i < sStepPairRecordCount; ++i)
	{
		const StepPairRecord &r = sStepPairRecords[i];
		// 记录按叶子指针对存（窄相位按叶子派发），这里把属于这个 body pair 的所有
		// (墙叶 × 撞击叶) 组合的两侧账各自求和、格号各自拼接。
		const uint32_t v1 = sLeafIndexLookup(r.mShape1);
		const uint32_t v2 = sLeafIndexLookup(r.mShape2);
		const bool side1 = (v1 & 0xFFFFu) != 0 && (v2 >> 16) != 0;
		const bool side2 = (v1 >> 16) != 0 && (v2 & 0xFFFFu) != 0;
		if (!side1 && !side2)
			continue;

		// 墙侧（outCells / outStrength）
		float s1 = side1? r.mStrength1 : r.mStrength2;
		uint count1 = side1? r.mCount1 : r.mCount2;
		const uint32_t *cells1 = side1? r.mCells1 : r.mCells2;
		// ★ 墙侧叶子恒在 wallLeaves 里（side1 时 = mShape1，side2 时 = mShape2）。
		//   旧版写成 "side1 ? wallLeaves : punchLeaves"：side2 方向的记录拿墙叶去撞击叶
		//   表里查，查不到 ⇒ 下标兜底 0 ⇒ 整批格号映射进 chunk 0（"没被撞到的位置也破"）。
		//   ★ 下标与这条记录的格号无关 ⇒ 每条记录只查一次，不能放进下面逐格循环里。
		//   旧写法对**每一颗格**都线性扫一遍叶子表（128 叶 × 上万格 ≈ 百万次比较/回调），
		//   是 Jolt/Step 里 C# 回调段的大头；现在整张表建成哈希索引，一条记录两次 O(1)。
		const uint16_t leaf1_idx = outLeafIdx != nullptr? uint16_t(((side1? v1 : v2) & 0xFFFFu) - 1u) : uint16_t(0);
		strength += s1;
		if (outCells != nullptr && inCapacity > 0)
		{
			if (count1 > inCapacity - copied)
				count1 = inCapacity - copied;	// 只拷能放下的部分：上层宁可少挖几格，也不要越界
			memcpy(outCells + copied, cells1, count1 * sizeof(uint32_t));
			if (outLeafIdx != nullptr)
				for (uint c = 0; c < count1; ++c)
					outLeafIdx[copied + c] = leaf1_idx;
			copied += count1;
		}

		// 撞击侧（outCells2 / outStrength2）—— 动×动 / 撞体自身判破用
		// ★ 同上：叶下标恒从 punchLeaves 那一半取（对称修法）。
		float s2 = side1? r.mStrength2 : r.mStrength1;
		uint count2 = side1? r.mCount2 : r.mCount1;
		const uint32_t *cells2 = side1? r.mCells2 : r.mCells1;
		const uint16_t leaf2_idx = outLeafIdx2 != nullptr? uint16_t((((side1? v2 : v1) >> 16) & 0xFFFFu) - 1u) : uint16_t(0);
		strength2 += s2;
		if (outCells2 != nullptr && inCapacity2 > 0)
		{
			if (count2 > inCapacity2 - copied2)
				count2 = inCapacity2 - copied2;
			memcpy(outCells2 + copied2, cells2, count2 * sizeof(uint32_t));
			if (outLeafIdx2 != nullptr)
				for (uint c = 0; c < count2; ++c)
					outLeafIdx2[copied2 + c] = leaf2_idx;
			copied2 += count2;
		}
	}
	if (outStrength != nullptr)
		*outStrength = strength;
	if (outStrength2 != nullptr)
		*outStrength2 = strength2;
	if (outCount2 != nullptr)
		*outCount2 = copied2;
	return copied;
}

/// 逐格碰撞结果的转发器：每产出一个接触结果就把**那一格**的材质强度加进所属的
/// 形状对记录（sStepPairClaim 占位的，单写者），然后原样转给下游 collector。
/// 沿用的是 Jolt 自己的 InternalEdgeRemovingCollector 的包装写法。
class VoxelStrengthSumCollector final : public CollideShapeCollector
{
public:
	// ★ inVoxel1 / inVoxel2：两个 shape 各自的 VoxelShape 指针（非体素侧传 null）。
	//   体素×体素时两个都给 ⇒ 每个 hit 两侧各解各的格号、各记各的账；
	//   体素×凸时只给一侧 ⇒ 另一侧保持 0。
	// ★ inRecord：sStepPairClaim 占位的形状对记录（表满时为 null ⇒ 只转发不记账）。
	VoxelStrengthSumCollector(CollideShapeCollector &inChained, const VoxelShape *inVoxel1, const VoxelShape *inVoxel2, StepPairRecord *inRecord) :
		CollideShapeCollector(inChained),
		mChained(inChained),
		mVoxel1(inVoxel1),
		mVoxel2(inVoxel2),
		mRecord(inRecord)
	{
	}

	virtual void			Reset() override
	{
		// 记录在 sStepPairClaim 占位时已清零；这里不再清（同对重复进来继续累加同一份）
		CollideShapeCollector::Reset();
	}

	virtual void			OnBody(const Body &inBody) override
	{
		mChained.OnBody(inBody);
	}

	virtual void			OnBodyEnd() override
	{
		mChained.OnBodyEnd();
	}

	virtual void			AddHit(const CollideShapeResult &inResult) override
	{
		// 各侧各解各的格号；非体素侧为 null（记录占位时清过零 ⇒ 保持 0，不会读到陈旧值）
		// ★ 两侧都必须判空：sCollideConvexVsVoxel（凸体在 shape1）传的是 (null, voxel)，
		//   CharacterVirtual 的脚下查询走的正是这条 —— 这里不解引用空指针。
		if (mVoxel1 != nullptr && mRecord != nullptr)
		{
			mRecord->mStrength1 += mVoxel1->GetContactStrength(inResult.mSubShapeID1);
			{
				uint cx, cy, cz;
				mVoxel1->DecodeSubShapeID(inResult.mSubShapeID1, cx, cy, cz);
				sRecordCell(mRecord, true, sPackCell(cx, cy, cz));
			}
		}
		if (mVoxel2 != nullptr && mRecord != nullptr)
		{
			mRecord->mStrength2 += mVoxel2->GetContactStrength(inResult.mSubShapeID2);
			{
				uint cx, cy, cz;
				mVoxel2->DecodeSubShapeID(inResult.mSubShapeID2, cx, cy, cz);
				sRecordCell(mRecord, false, sPackCell(cx, cy, cz));
			}
		}

		// 原样转发（下游可能是 AllHit，也可能是约减 collector —— 一律不改它的语义）
		mChained.SetContext(GetContext());
		mChained.AddHit(inResult);
		UpdateEarlyOutFraction(mChained.GetEarlyOutFraction());
	}

private:
	CollideShapeCollector &	mChained;
	const VoxelShape *		mVoxel1;
	const VoxelShape *		mVoxel2;
	StepPairRecord *		mRecord;
};


/// ★★★ 这 256 项是**静态存储、不是 new 出来的**，但 PhysicsMaterial 派生自 RefTarget
///   （引用计数）—— 而 Shape::GetMaterial() 的返回值会被上层塞进 RefConst<>，
///   最典型的就是角色控制器：CharacterVirtual::UpdateSupportingContact 里的
///       mGroundMaterial = best_contact->mMaterial;   // mGroundMaterial 是 RefConst<PhysicsMaterial>
///   引用计数从 0 起的话，RefConst 那次 AddRef 让它变成 1，**下一次 Release 就会**
///       if (mRefCount.fetch_sub(1) == 1) delete this;
///   ⇒ delete 一个静态对象 ⇒ operator delete → free(非堆地址) ⇒ ntdll 报
///   STATUS_HEAP_CORRUPTION (0xC0000374)，进程当场消失（无托管异常、无 crash 报告）。
///   现象：一站在体素地形上、换个支撑接触就闪退；关掉逐体素物理材质则正常。
///   SetEmbedded() 是 Jolt 给"栈上/静态、生命周期不由引用计数管"的对象准备的标记：
///   给计数加上哨兵 cEmbedded，Release 永远走不到 1 ⇒ 永不 delete
///   （Jolt 自己也这么用：BodyManager 的默认形状、VehicleCollisionTester 的栈上形状）。
///   **下面这个初始化不能删。**
static const bool sVoxelMaterialsEmbedded = []() {
	for (VoxelMaterial &m : sVoxelMaterials)
		m.SetEmbedded();
	return true;
}();

/// 当前生效的材质表（写时复制，见 sVoxelMaterials 的说明）。初始指向上面那张静态默认表：
/// 没调过 sSetMaterialTable() 时它给出默认值（摩擦 0.5 / 弹性 0）。指针本身是原子常量初始化，
/// 所以任何静态初始化顺序下都有效。
static atomic<const VoxelMaterial *> sVoxelMaterialTable { sVoxelMaterials };

/// 取当前生效的材质表（读侧唯一入口）。acquire：与 sSetMaterialTable 的 release 发布配对，
/// 保证看到新指针时也看得到表里已经填好的 friction / restitution。
static inline const VoxelMaterial *sGetVoxelMaterialTable()
{
	return sVoxelMaterialTable.load(memory_order_acquire);
}

/// 判断射线方向的某个分量是否可以当成 0 来对待（退化成"与该轴平行"）
static constexpr float cDirectionEpsilon = 1.0e-20f;

/// 计算格子下标范围时的浮点容差：正好落在格子边界上的坐标归给"上面"那一格，
/// 这样相邻格子只会在真正重叠时才产生接触，不会因为浮点误差互相重复计数。
static constexpr float cCellEpsilon = 1.0e-4f;

// ---- 4 车道 floor / ceil：Vec4::ToInt() 是 cvtps2dq 的**向零截断**版，负的非整数要修正 ----
//   floor(x)：截断后若比 x 大（说明 x 是负的非整数），减 1
//   ceil(x) ：截断后若比 x 小（说明 x 是正的非整数），加 1
// 有了它们，4 个遍历格的候选下标区间可以一次算完，替代每格 3 次 std::floor / std::ceil。
static JPH_INLINE Vec4 sFloor4(Vec4Arg inV)
{
	Vec4 t = inV.ToInt().ToFloat();
	return Vec4::sSelect(t, t - Vec4::sReplicate(1.0f), Vec4::sLess(inV, t));
}

static JPH_INLINE Vec4 sCeil4(Vec4Arg inV)
{
	Vec4 t = inV.ToInt().ToFloat();
	return Vec4::sSelect(t, t + Vec4::sReplicate(1.0f), Vec4::sGreater(inV, t));
}

// 体素数据版本号（**全局唯一**，每次 Recalculate() 换一个新值）= "形状身份" + "数据世代"：
// 让按版本号缓存的东西（调试几何等）在数据变化后一次性失效。
static atomic<uint> sNextVoxelDataVersion { 1 };

/// 把世界空间的一个轴对齐盒，变换到体素形状的**未缩放局部空间**（GetVoxelCenter 所在的空间）。
///
/// 做法是先把 8 个角点变换到"含缩放的局部空间"，再按轴除以缩放。对旋转过的形状这会得到一个
/// 保守（略大）的包围盒 —— 这是刻意的：多遍历几个空格子只是浪费一点时间，漏掉格子会丢碰撞。
static AABox sWorldBoxToVoxelLocal(const AABox &inWorldBox, Mat44Arg inVoxelTransform, Vec3Arg inVoxelScale)
{
	Mat44 world_to_voxel = inVoxelTransform.Inversed();

	// 8 个角点 -> 含缩放的局部空间
	AABox scaled_local;
	for (uint i = 0; i < 8; ++i)
	{
		Vec3 corner(
			(i & 1) != 0? inWorldBox.mMax.GetX() : inWorldBox.mMin.GetX(),
			(i & 2) != 0? inWorldBox.mMax.GetY() : inWorldBox.mMin.GetY(),
			(i & 4) != 0? inWorldBox.mMax.GetZ() : inWorldBox.mMin.GetZ());
		scaled_local.Encapsulate(world_to_voxel * corner);
	}

	// 再除以缩放 -> 未缩放局部空间
	Vec3 inv_scale(
		1.0f / inVoxelScale.GetX(),
		1.0f / inVoxelScale.GetY(),
		1.0f / inVoxelScale.GetZ());
	AABox result;
	result.Encapsulate(scaled_local.mMin * inv_scale);
	result.Encapsulate(scaled_local.mMax * inv_scale);
	return result;
}

/// 单位立方体的角点数量
static constexpr int cNumUnitCubePoints = 8;

/// 单位立方体（边长 1、中心在原点）的 8 个角点，供 GetSubmergedVolume 使用
static const Vec3 sUnitCubePoints[cNumUnitCubePoints] =
{
	Vec3(-0.5f, -0.5f, -0.5f),
	Vec3( 0.5f, -0.5f, -0.5f),
	Vec3(-0.5f,  0.5f, -0.5f),
	Vec3( 0.5f,  0.5f, -0.5f),
	Vec3(-0.5f, -0.5f,  0.5f),
	Vec3( 0.5f, -0.5f,  0.5f),
	Vec3(-0.5f,  0.5f,  0.5f),
	Vec3( 0.5f,  0.5f,  0.5f)
};

/// 单位立方体的 6 个面（顶点按"从外面看逆时针"排列），最后一列是"这个面用到了哪些角点"的位掩码，
/// 用来跳过包含参考点（最低点）的面 —— 那种面贡献的体积恒为 0。
struct UnitCubeFace
{
	int				mIndices[4];
	int				mMask;
};
static const UnitCubeFace sUnitCubeFaces[] =
{
	{ { 0, 2, 3, 1 }, (1 << 0) | (1 << 2) | (1 << 3) | (1 << 1) },
	{ { 4, 6, 2, 0 }, (1 << 4) | (1 << 6) | (1 << 2) | (1 << 0) },
	{ { 4, 5, 7, 6 }, (1 << 4) | (1 << 5) | (1 << 7) | (1 << 6) },
	{ { 1, 3, 7, 5 }, (1 << 1) | (1 << 3) | (1 << 7) | (1 << 5) },
	{ { 2, 6, 7, 3 }, (1 << 2) | (1 << 6) | (1 << 7) | (1 << 3) },
	{ { 0, 1, 5, 4 }, (1 << 0) | (1 << 1) | (1 << 5) | (1 << 4) }
};

/// 求"某一个格子（边长 1 的立方体被 inCellTransform 变换到目标空间）与水面相交部分"的体积和浮心。
/// ★ outSubmergedVolume 是**目标空间里的绝对体积**，不是 0..1 的比例：
///   PolyhedronSubmergedVolumeCalculator::GetResult 直接给变换后的体积（ConvexShape::GetSubmergedVolume
///   也是这么用的），而 inCellTransform 里已经含了体素边长和形状缩放 ⇒ 单位立方体被放大成了真实格子大小。
///   inCellVolume 只在"整格都在水下"的早退分支里用得到（那时算不出体积，直接给全格）。
/// 返回 false 表示"整格都在水面之上"（什么都不用累加）。
static bool sGetCellSubmergedVolume(Mat44Arg inCellTransform, float inCellVolume, const Plane &inSurface, float &outSubmergedVolume, Vec3 &outCenterOfBuoyancy JPH_IF_DEBUG_RENDERER(, RVec3Arg inBaseOffset))
{
	PolyhedronSubmergedVolumeCalculator::Point buffer[cNumUnitCubePoints];
	PolyhedronSubmergedVolumeCalculator calculator(inCellTransform, sUnitCubePoints, sizeof(Vec3), cNumUnitCubePoints, inSurface, buffer JPH_IF_DEBUG_RENDERER(, inBaseOffset));

	// 早退：整格都在水面之上
	if (calculator.AreAllAbove())
	{
		outSubmergedVolume = 0.0f;
		outCenterOfBuoyancy = Vec3::sZero();
		return false;
	}

	// 早退：整格都在水面之下。正常情况下调用方已经用 cell_radius 提前判掉了，
	// 只有"格子的角点刚好压在水面上"这种退化情形才会走到这里。
	if (calculator.AreAllBelow())
	{
		outSubmergedVolume = inCellVolume;
		outCenterOfBuoyancy = inCellTransform.GetTranslation();
		return true;
	}

	// 部分浸没：把所有面（除去那些用到参考点、贡献恒为 0 的面）加进去
	int reference_point_bit = 1 << calculator.GetReferencePointIdx();
	for (const UnitCubeFace &face : sUnitCubeFaces)
		if ((face.mMask & reference_point_bit) == 0)
		{
			calculator.AddFace(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
			calculator.AddFace(face.mIndices[0], face.mIndices[2], face.mIndices[3]);
		}

	calculator.GetResult(outSubmergedVolume, outCenterOfBuoyancy);
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////
// 手写体素窄相位用到的小工具
//////////////////////////////////////////////////////////////////////////////////////////

/// 一个有朝向的盒子（OBB）：可以是"整个体素网格"，也可以是"其中一格"。
/// mAxis 是三条**单位化**的轴，mHalf 是沿这三条轴各自的半长（世界单位）。
struct VoxelOBB
{
	Vec3	mCenter;
	Vec3	mAxis[3];
	Vec3	mHalf;

	/// 哪些**网格轴方向是"延续到邻区块的假边界"**（bit k = mAxis[k]，1 = 那边不是真实的外表面）。
	/// 地形是一大片相邻区块拼出来的：对单个区块来说，±x / ±z 那四个侧面接的是隔壁区块的实心地形，
	/// 不是空气 ⇒ 不能把"沿 x 走出这个区块"当成有效分离方向（见 sObbMinSeparationOverlap）。
	/// ★ 位是**轴级**的（± 两个方向共用一位）：重叠区附近的洞口检测（sRefineBorderMask）会把
	///   "边界层真被挖空"的方向从位里放掉 —— 只要 ± 任一方向探到洞口，整条轴重新成为合法候选
	///   （这里不做方向级门控：
	///   深穿透构型下盒心符号与真实出口可能相反，方向级判定会查错方向）。
	uint	mContinuousMask = 0;
};

/// 构造"整个体素网格"的 OBB。
/// 缩放作用在形状自己的局部轴上，所以第 i 根轴上的半长 = 0.5 * size[i] * voxelSize * |scale[i]|；
/// 旋转只改变轴的方向（轴取变换矩阵 3x3 部分的列，平移单独存）。
/// inContinuousMask = **轴级** 3 位延续掩码（bit k = 第 k 条网格轴，由 GetContinuousMask()
/// 经 sRefineBorderMask 按重叠区附近的实际占用细化而来）。
static VoxelOBB sGetGridOBB(int inSizeX, int inSizeY, int inSizeZ, float inVoxelSize, Vec3Arg inScale, Mat44Arg inCenterOfMassTransform, uint inContinuousMask = 0)
{
	VoxelOBB obb;
	obb.mCenter = inCenterOfMassTransform.GetTranslation();
	for (int i = 0; i < 3; ++i)
		obb.mAxis[i] = inCenterOfMassTransform.GetColumn3(i).Normalized();
	obb.mHalf = Vec3(
		0.5f * float(inSizeX) * inVoxelSize * abs(inScale.GetX()),
		0.5f * float(inSizeY) * inVoxelSize * abs(inScale.GetY()),
		0.5f * float(inSizeZ) * inVoxelSize * abs(inScale.GetZ()));
	obb.mContinuousMask = inContinuousMask;
	return obb;
}

/// 按重叠区附近的**实际占用**细化延续掩码：真实的洞口方向把轴位清掉，轴重新成为合法候选。
///
/// 输入/输出都是 GetContinuousMask() 那种 3 位轴掩码（bit k = 第 k 条网格轴那边是延续的），
/// 输出只是"掩掉了判实为假边界之后剩下的位"，编码不变（± 两方向共用一位，见 VoxelOBB）。
///
/// ★★ 为什么要细化：掩码的本意是"这个方向外头还是实心地形，沿它分离无效"。这只在**重叠区
///    附近沿该方向真的全是实心**时成立 —— 挖洞 / 碎块撕走之后，洞口方向是真实出口；再按
///    整条轴丢弃，贴洞壁的碎块就只剩竖直候选轴，被洞壁/洞底来回顶（抖动 / 往深顶）。
///
/// ★ 位是轴级的、不做方向级门控：深穿透 / 包含构型下，盒心连线的符号与真实出口可能相反
///   （方块楔进洞壁时盒心在洞壁一侧，SAT 的 A→B 指向墙里，而正确出口是穿回洞口），方向要
	///   所以只要 ± 任一方向探到洞口，整条轴放行。
///
/// 两条洞口规则（保守优先：判不准就保留位 = 旧行为；交集为空（推测性接触）⇒ 不细化）：
///   规则 1（外边界面）：重叠区贴到区块边界面（距边 ≤ 2 格）时，查**边界层条带**（该轴最后
///     一层格 × 重叠区切向范围）—— 全空 ⇒ 这个面在重叠区段是真洞口（物体直接穿面而出）。
///   规则 2（内部腔体）：从重叠区沿该方向**向外逐层探链**（≤ cHoleProbeCells 层，切向范围 =
///     重叠区）—— 任一层碰到空气 ⇒ 重叠区外侧就是洞/腔（碎块从地形内部撕走的情形）。
///     探链出网格 = 一路实心到区块边 = 延续成立 ⇒ 保留位。
///   切向范围刻意**不**外扩（物体只能扫过它自己的截面；外扩会把洞口旁边的一圈实心格卷进来，
///   把真洞口误判成延续），并夹紧到每轴 ≤ cHoleProbeSpanCells 格封顶最坏开销。
static uint sRefineBorderMask(const VoxelShape *inVoxel, Mat44Arg inTransform, Vec3Arg inScale, const AABox &inMeetWorld)
{
	constexpr int cHoleProbeCells = 8;			// 规则 2 的探链深度（格）：洞壁距重叠区不会太远
	constexpr int cHoleProbeSpanCells = 32;		// 切向范围每轴封顶（格）：封住大物体 vs 地形的最坏开销

	const uint axis_mask = inVoxel->GetContinuousMask();
	if (axis_mask == 0)
		return 0;
	uint refined = axis_mask;

	// 交集为空（推测性接触）⇒ 不干预
	if (inMeetWorld.mMin.GetX() > inMeetWorld.mMax.GetX()
		|| inMeetWorld.mMin.GetY() > inMeetWorld.mMax.GetY()
		|| inMeetWorld.mMin.GetZ() > inMeetWorld.mMax.GetZ())
		return refined;

	const float vs = inVoxel->GetVoxelSize();
	const int size[3] = { inVoxel->GetSizeX(), inVoxel->GetSizeY(), inVoxel->GetSizeZ() };
	const AABox local = sWorldBoxToVoxelLocal(inMeetWorld, inTransform, inScale);
	// 局部米 -> 格下标**连续值**（floor/ceil 在使用处做，
	// 用整数截断自实现，避免引入 <cmath>；连续值恒 >= 0，截断即 floor）
	auto to_cell = [&](const Vec3 &inV, int inAxis) -> float
	{
		float c = inAxis == 0? inV.GetX() : (inAxis == 1? inV.GetY() : inV.GetZ());
		return c / vs + 0.5f * float(size[inAxis]);
	};

	for (int k = 0; k < 3; ++k)
	{
		if ((axis_mask & (1u << k)) == 0)
			continue;
		const float lo = to_cell(local.mMin, k);
		const float hi = to_cell(local.mMax, k);
		// 重叠区实际覆盖的格 = floor(lo) .. ceil(hi)-1（[a,b] 覆盖规则；hi 正好落在格边界上时
		// 归下一格 -- 不多测那一格，否则洞口旁边的一圈实心格会被卷进探针把真洞口误判成延续），
		// 再夹紧进网格并按 cHoleProbeSpanCells 封顶。
		const int a[2] = { (k + 1) % 3, (k + 2) % 3 };
		int span_lo[2], span_hi[2];
		for (int t = 0; t < 2; ++t)
		{
			float slo = to_cell(local.mMin, a[t]), shi = to_cell(local.mMax, a[t]);
			int clo = int(slo);													// floor
			int chi = int(shi) < shi? int(shi) + 1 : int(shi);					// ceil
			chi = chi - 1 >= clo? chi - 1 : clo;								// ceil - 1，退化区间兜底
			int center = (clo + chi) / 2;
			clo = max(clo, center - (cHoleProbeSpanCells / 2));
			chi = min(chi, center + (cHoleProbeSpanCells / 2) - 1);
			span_lo[t] = Clamp(clo, 0, size[a[t]] - 1);
			span_hi[t] = Clamp(chi, 0, size[a[t]] - 1);
		}
		// 在切向范围内查一格
		auto any_solid_in_span = [&](int inK) -> bool
		{
			int c[3];
			c[k] = inK;
			for (int u = span_lo[0]; u <= span_hi[0]; ++u)
			{
				c[a[0]] = u;
				for (int w = span_lo[1]; w <= span_hi[1]; ++w)
				{
					c[a[1]] = w;
					if (inVoxel->IsSolid(c[0], c[1], c[2]))
						return true;
				}
			}
			return false;
		};

		bool hole = false;
		// 规则 1：贴到 +k 边界面，且边界层条带全空 ⇒ 穿面而出的洞口
		if (hi > float(size[k]) - 2.0f && !any_solid_in_span(size[k] - 1))
			hole = true;
		// 规则 1：−k 侧同理
		if (!hole && lo < 1.0f && !any_solid_in_span(0))
			hole = true;
		// 规则 2：从重叠区两侧向外逐层探链（≤ cHoleProbeCells 层），任一层碰到空气 = 腔体
		if (!hole)
		{
			int first_beyond[2];						// 重叠区外（沿 ±k）的第一层格
			first_beyond[0] = int(hi) < hi? int(hi) + 1 : int(hi);			// ceil(hi)
			first_beyond[1] = int(lo) - 1;									// floor(lo) - 1
			for (int d = 0; d < 2 && !hole; ++d)
			{
				int sign = d == 0? 1 : -1;
				int ck = first_beyond[d];
				for (int step = 0; step < cHoleProbeCells; ++step, ck += sign)
				{
					if (ck < 0 || ck >= size[k])
						break;						// 出网格：一路实心到边 = 延续成立，保留位
					if (!any_solid_in_span(ck))
					{
						hole = true;				// 重叠区外侧是空气 = 腔体/洞口
						break;
					}
				}
			}
		}
		if (hole)
			refined &= ~(1u << k);
	}
	return refined;
}

/// 实心格集合的世界 AABB（8 角点经"质心变换 x 形状缩放"变换，与 sWorldBoxToVoxelLocal 互逆）。
/// 派生统计无效（从未 EnsureDerived）时回退整网格盒 —— 保守，绝不会因此漏接触。
/// 窄相位用它圈"双方实心格可能重叠的区域"：整网格 OBB 含全部空格，大而空的物体悬在地形
/// 上方时会把正下方几万格空格/深层格整片圈进遍历范围（离线实测 0 接触也要几百 us 到几 ms）。
	AABox VoxelShape::sSolidAABBWorld(const VoxelShape *inVoxel, Mat44Arg inCenterOfMassTransform, Vec3Arg inScale)
{
	const float vs = inVoxel->mVoxelSize;
	const float half[3] = { inVoxel->mHalfExtent.GetX(), inVoxel->mHalfExtent.GetY(), inVoxel->mHalfExtent.GetZ() };
	const float size[3] = { float(inVoxel->mSizeX), float(inVoxel->mSizeY), float(inVoxel->mSizeZ) };
	const int solid_min[3] = { inVoxel->GetSolidMin(0), inVoxel->GetSolidMin(1), inVoxel->GetSolidMin(2) };
	const int solid_max[3] = { inVoxel->GetSolidMax(0), inVoxel->GetSolidMax(1), inVoxel->GetSolidMax(2) };
	// ★★★ 全空形状（挖光了 / 哨兵 min>max）必须返回**空盒**。旧版照常算 lo/hi，得到
	//     lo > hi 的反盒，下面 8 角点 Encapsulate 又把它混回整网格大小的合法盒 ⇒ 空块
	//     照样报出一个覆盖整网格的「实心 AABB」，和地形相交后窄相位整片空跑。
	for (int k = 0; k < 3; ++k)
		if (solid_min[k] > solid_max[k])
			return AABox();
	float lo[3], hi[3];
	for (int k = 0; k < 3; ++k)
	{
		// 格 c 在该轴上占据 [c * s - H, (c + 1) * s - H]（未缩放局部空间）；
		// 派生统计无效时 solid_min/solid_max 回退整张网格（0 .. size-1）
		lo[k] = float(solid_min[k]) * vs - half[k];
		hi[k] = (float(solid_max[k]) + 1.0f) * vs - half[k];
	}
	const Mat44 m = inCenterOfMassTransform * Mat44::sScale(inScale);
	AABox box;
	for (int i = 0; i < 8; ++i)
		box.Encapsulate(m * Vec3((i & 1) != 0? hi[0] : lo[0],
		                         (i & 2) != 0? hi[1] : lo[1],
		                         (i & 4) != 0? hi[2] : lo[2]));
	return box;
}


/// 两个 OBB 的**整体分离量**（候选轴 = 两个盒子各自的 3 根面轴 + 9 根边叉积轴，共 15 根，
/// 在全部候选轴上取最小重叠 —— 注意这只产生一个标量，不再产出任何"轴"）。
///
/// 返回值：正数 = 最小重叠这么深，0 = 刚好贴合，负数 = 分离了这么多。outSecondOverlap = 次小。
///
/// ★ 它唯一的职责是早退（"整体都分离了 ⇒ 不可能有接触"）与诊断（sLastGridOverlap）。
///   全局轴**只**来自 sCollideVoxelGrids 1c) 的两条规则（盒心 − 接触区中心 / 盒心差），
///   这里的遍历过程不向外面交任何方向。保留 15 轴遍历而不是换简单 AABB 间隙，是因为
///   候选轴剔除带连续掩码语义（地形侧壁不算有效出口），换掉会改变早退边界。
static float sObbMinSeparationOverlap(const VoxelOBB &inA, const VoxelOBB &inB, float &outSecondOverlap)
{
	Vec3 delta = inB.mCenter - inA.mCenter;			// 从 A 的盒心指向 B 的盒心

	// 收集候选轴并顺手去重：两个网格朝向一致时，9 根叉积轴全部与面轴重合，
	// 去重后候选数从 15 降到 3，省掉 12 次投影计算。
	Vec3 axes[15];
	int num_axes = 0;

	// ★ 与某个"延续方向"平行的候选轴要**整条丢弃**（见 VoxelOBB::mContinuousMask）。
	//   地形区块的 ±x / ±z 四个侧面接的是隔壁区块的实心地形 —— 沿那些方向"分离"只是把物体推进
	//   另一个区块，等于没分离；而它们的行程恰恰最短（区块 8 m，物体贴到侧壁时行程趋近 0）
	//   ⇒ SAT 会稳定地选择"把物体推出区块"：静置时被横向推来推去，埋进地形时也不往上顶。
	//   注意这是**双方的**约束：哪怕只有地形那一侧标了延续，这个方向对这对形状就不是有效出口。
	//   ★ 掩码是轴级的，且已经过 sRefineBorderMask 细化：重叠区附近的真实洞口会把位清掉，
	//     同一根轴重新成为合法候选。这里刻意不做方向级门控 —— 深穿透/包含构型下盒心连线的
	//     符号与真实出口可能相反，按方向掩会查错方向。
	//   ★ 平行判据用 0.98（≈11°）而不是 0.9999（0.8°）：两个网格只是**稍微**转了个角度时
	//     （实测方块在斜坡上侧倾 2.5°），9 根叉积轴会给出"几乎沿着格轴、偏几度"的轴 ——
	//     (0.010,-0.043,-0.999) 就是实测样本。它沿 z 前进 1 m 只偏 4 cm，穿的**仍然是**
	//     隔壁区块的实心地形 ⇒ 语义上就是"沿这个连续方向分离"，必须一起丢掉。
	//     真正需要保留的斜轴（例如 45° 的斜面/挖洞）dot ≈ 0.7，离 0.98 很远。
	auto is_continuous = [&inA, &inB](Vec3Arg inAxis)
	{
		for (int k = 0; k < 3; ++k)
		{
			if ((inA.mContinuousMask & (1u << k)) != 0 && abs(inA.mAxis[k].Dot(inAxis)) > 0.98f)
				return true;
			if ((inB.mContinuousMask & (1u << k)) != 0 && abs(inB.mAxis[k].Dot(inAxis)) > 0.98f)
				return true;
		}
		return false;
	};

	auto add_axis = [&axes, &num_axes, &is_continuous](Vec3Arg inAxis)
	{
		float length = inAxis.Length();
		// ★ 退化叉积必须跳过：两根轴都归一化时 |cross| = sin(夹角)，1e-3 对应 0.06°。
		//   原来的 1e-6 太宽松 —— 差 0.06° 以内的叉积会被归一化成一根**方向由数值噪声决定**
		//   的伪轴，它的投影半径/重叠量没有任何几何意义，却可能因为"算出来更小"被 SAT 选中。
		if (length < 1.0e-3f)
			return;
		Vec3 axis = inAxis / length;
		if (is_continuous(axis))
			return;
		for (int i = 0; i < num_axes; ++i)
			if (abs(axes[i].Dot(axis)) > 0.9999f)	// 已经有一根同向（或反向）的轴了
				return;
		axes[num_axes++] = axis;
	};

	for (int i = 0; i < 3; ++i)
	{
		add_axis(inA.mAxis[i]);
		add_axis(inB.mAxis[i]);
	}
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			add_axis(inA.mAxis[i].Cross(inB.mAxis[j]));

	// 保底：两边都掩到一根候选都不剩（例如两个地形区块互撞、且朝向一致时叉积轴全部退化）
	if (num_axes == 0)
		for (int i = 0; i < 3; ++i)
		{
			add_axis(inA.mAxis[i]);
			add_axis(inB.mAxis[i]);
		}

	float best_overlap = FLT_MAX;
	float second_overlap = FLT_MAX;

	for (int i = 0; i < num_axes; ++i)
	{
		const Vec3 &axis = axes[i];

		// 两个盒子在这根轴上的投影半径
		float radius_a = 0.0f, radius_b = 0.0f;
		for (int k = 0; k < 3; ++k)
		{
			radius_a += abs(axis.Dot(inA.mAxis[k])) * inA.mHalf[k];
			radius_b += abs(axis.Dot(inB.mAxis[k])) * inB.mHalf[k];
		}

		float distance = axis.Dot(delta);
		float overlap = radius_a + radius_b - abs(distance);
		if (overlap < best_overlap)
		{
			second_overlap = best_overlap;
			best_overlap = overlap;
			// 统一指向"从 A 到 B"的那一侧，接触法线才有确定的含义
		}
		else if (overlap < second_overlap)
			second_overlap = overlap;
	}

	// ★★【已删：接触区域厚度改判 + 嵌套门控】
	//   这里原来用 meet（实心 AABB 交集）的 8 个角点在各候选轴上投影取厚度，唯一最薄才改判轴，
	//   并且要求"胜出轴上双方投影区间嵌套"才启用。那本质上是**给"包含"这一种构型打的补丁**：
	//   体素能拼出任意形状（包含 / 斜坡 / 缝隙 / 深穿透 / 悬挑 / 薄墙……），按构型分支迟早烂掉，
	//   而且这些分支之间还会互相打架。
	//   现在 SAT 只负责**整体分离量**（早退 + 诊断），轴本身由调用方一条与构型无关的公式给出：
	//       axis = 盒心(shape2) − 接触区中心，退化时 = 盒心(shape2) − 盒心(shape1)
	//   （见 sCollideVoxelGrids 的 1c) 节）。
	outSecondOverlap = second_overlap;
	return best_overlap;
}

/// 一个盒子（单位盒经 inTransform 变换、按 inScale 缩放之后）沿 inAxis 的投影半径。
/// 就是"盒子的支撑点在这根轴上能伸出多远"，SAT 和逐格穿透深度都要用它。
static float sGetProjectionRadius(const Mat44 &inTransform, Vec3Arg inScale, Vec3Arg inAxis)
{
	return 0.5f * (abs(inAxis.Dot(inTransform.GetColumn3(0) * inScale.GetX()))
				 + abs(inAxis.Dot(inTransform.GetColumn3(1) * inScale.GetY()))
				 + abs(inAxis.Dot(inTransform.GetColumn3(2) * inScale.GetZ())));
}

/// 一次查询里最多产出的体素接触数。
/// 逐格接触的数量正比于重叠区的格子数，两个大网格深穿透时会非常多；超过这个数就不再产接触，
/// 免得一次窄相位就吃掉几毫秒。
/// ★ 只限制**接触点**，不再限制遍历：撞击记账（inRecord != nullptr）会把重叠格扫完（见 keep_scanning），
///   因为破坏要的是"这一击撞到的完整格集合"，截断它就会出现"洞挖不干净 ⇒ 越撞越穿"。
int VoxelShape::sMaxContactPoints = 1024;

/// 见 VoxelShape.h 里 sUseSphereContacts 的说明。默认开：体素 vs 其它形状每格用内切球。
bool VoxelShape::sUseSphereContacts = true;

/// 见 VoxelShape.h 里 sInteriorSkipPairs 的说明。默认 2 = 任一侧是内部格（6 邻全实心）就跳过该格对。
int VoxelShape::sInteriorSkipPairs = 2;

/// 见 VoxelShape.h 里 sIterateSideOverride 的说明。0 = 自动（遍历重叠区格数少的一侧）。
int VoxelShape::sIterateSideOverride = 0;

/// 见 VoxelShape.h 里 sRotatedFastPath 的说明。默认开：只改旋转网格的枚举方式，接触集合不变。
int VoxelShape::sRotatedFastPath = 1;

/// 窄相位统计（编辑器调试面板）：遍历过的格数 / 其中实心进入接触计算的格数。relaxed 就够（诊断用）。
std::atomic<uint64> VoxelShape::sStatVisitedCells = 0;
std::atomic<uint64> VoxelShape::sStatScanCells = 0;
std::atomic<uint64> VoxelShape::sStatColumnCells = 0;
std::atomic<uint64> VoxelShape::sStatEnsureDerivedCalls = 0;
std::atomic<uint64> VoxelShape::sStatEnsureDerivedNs = 0;
std::atomic<bool> VoxelShape::sHotStatsEnabled = true;
std::atomic<int> VoxelShape::sStatLastIterSide = 0;
int VoxelShape::sStatLastRange[6] = {0,0,0,0,0,0};
std::atomic<uint64> VoxelShape::sStatSolidCells = 0;
std::atomic<uint64> VoxelShape::sStatContacts = 0;
std::atomic<uint64> VoxelShape::sStatNarrowPhaseNs = 0;
std::atomic<uint64> VoxelShape::sStatNarrowPhaseCalls = 0;
std::atomic<uint64> VoxelShape::sStatProbes = 0;
std::atomic<uint64> VoxelShape::sStatNarrowPhaseMaxNs = 0;
std::mutex VoxelShape::sNarrowPhaseMaxMutex;
uint64 VoxelShape::sNarrowPhaseMaxVisited = 0;
uint64 VoxelShape::sNarrowPhaseMaxSolid = 0;
uint64 VoxelShape::sNarrowPhaseMaxProbes = 0;
uint64 VoxelShape::sNarrowPhaseMaxProbeNs = 0;
uint64 VoxelShape::sNarrowPhaseMaxWalkNs = 0;
uint64 VoxelShape::sNarrowPhaseMaxSubTests = 0;
uint64 VoxelShape::sNarrowPhaseMaxSubEmpty = 0;
uint64 VoxelShape::sNarrowPhaseMaxScan = 0;
int VoxelShape::sNarrowPhaseMaxRange[6] = {0,0,0,0,0,0};
std::atomic<uint32> VoxelShape::sDebugQueryStamp = 0;

void VoxelShape::TakeQueryStats(uint64 &outVisited, uint64 &outSolid, uint64 &outContacts, uint64 &outNarrowPhaseNs, uint64 &outNarrowPhaseCalls, uint64 &outProbes, uint64 &outNarrowPhaseMaxNs, uint64 &outMaxCallVisited, uint64 &outMaxCallSolid, uint64 &outMaxCallProbes, uint64 &outMaxCallProbeNs)
{
	outVisited = sStatVisitedCells.exchange(0, std::memory_order_relaxed);
	outSolid = sStatSolidCells.exchange(0, std::memory_order_relaxed);
	outContacts = sStatContacts.exchange(0, std::memory_order_relaxed);
	outNarrowPhaseNs = sStatNarrowPhaseNs.exchange(0, std::memory_order_relaxed);
	outNarrowPhaseCalls = sStatNarrowPhaseCalls.exchange(0, std::memory_order_relaxed);
	outProbes = sStatProbes.exchange(0, std::memory_order_relaxed);
	outNarrowPhaseMaxNs = sStatNarrowPhaseMaxNs.exchange(0, std::memory_order_relaxed);
	{
		std::lock_guard<std::mutex> lock(sNarrowPhaseMaxMutex);
		outMaxCallVisited = sNarrowPhaseMaxVisited; sNarrowPhaseMaxVisited = 0;
		outMaxCallSolid = sNarrowPhaseMaxSolid; sNarrowPhaseMaxSolid = 0;
		outMaxCallProbes = sNarrowPhaseMaxProbes; sNarrowPhaseMaxProbes = 0;
		outMaxCallProbeNs = sNarrowPhaseMaxProbeNs; sNarrowPhaseMaxProbeNs = 0;
	}
}

void VoxelShape::TakeScanStats(uint64 &outScanCells, uint64 &outColumnCells, int &outIterSide, int outRange[6])
{
	outScanCells = sStatScanCells.exchange(0, std::memory_order_relaxed);
	outColumnCells = sStatColumnCells.exchange(0, std::memory_order_relaxed);
	outIterSide = sStatLastIterSide.load(std::memory_order_relaxed);
	for (int k = 0; k < 6; ++k)
		outRange[k] = sStatLastRange[k];
}

void VoxelShape::TakeEnsureDerivedStats(uint64 &outCalls, uint64 &outNs)
{
	outCalls = sStatEnsureDerivedCalls.exchange(0, std::memory_order_relaxed);
	outNs = sStatEnsureDerivedNs.exchange(0, std::memory_order_relaxed);
}

void VoxelShape::TakeMaxCallScanStats(uint64 &outScan, uint64 &outWalkNs, uint64 &outSubTests, uint64 &outSubEmpty, int outRange[6])
{
	std::lock_guard<std::mutex> lock(sNarrowPhaseMaxMutex);
	outScan = sNarrowPhaseMaxScan; sNarrowPhaseMaxScan = 0;
	outWalkNs = sNarrowPhaseMaxWalkNs; sNarrowPhaseMaxWalkNs = 0;
	outSubTests = sNarrowPhaseMaxSubTests; sNarrowPhaseMaxSubTests = 0;
	outSubEmpty = sNarrowPhaseMaxSubEmpty; sNarrowPhaseMaxSubEmpty = 0;
	for (int k = 0; k < 6; ++k)
	{
		outRange[k] = sNarrowPhaseMaxRange[k];
		sNarrowPhaseMaxRange[k] = 0;
	}
}

/// RAII 计时：把一次窄相位调用的墙钟时间累进 sStatNarrowPhaseNs（诊断用，relaxed 即可）。
/// 同时维护"单次最大调用"快照（那次调用的 遍历/实心/探测 计数），用来定性
/// "总耗时 = 1 次独大还是均摊"以及"那次调用到底在干什么"。
/// sCallProbeLoopNs（thread_local）= 本线程当前调用里候选循环累计耗时，随快照一起带出，
/// 用来拆"探测循环 vs 调用内其它准备段"。
static thread_local uint64 sCallProbeLoopNs = 0;
/// 本线程当前调用里"遍历行走段"累计耗时（sVisitSolidVoxelsInBox + flush 批处理 + chunk 剪枝，
/// **含**其中的探测循环；快照里记录 walk - probe = 纯行走成本），随快照一起带出。
static thread_local uint64 sCallWalkNs = 0;
/// 本线程当前调用里"子块位图测试"计数（sVisitSolidVoxelsInBox 的 4^3 子块循环迭代数 = sweep 体积）
/// 与其中 sub == 0 的空跳数（空跳率决定"chunk 级摘要整块跳过"修法的收益），随快照一起带出。
static thread_local uint64 sCallSubTests = 0;
static thread_local uint64 sCallSubEmpty = 0;
/// 本线程最近一次查询的 range（与 sStatLastRange 同步写入；dtor 快照用它做**本调用**归属，
/// 全局的 sStatLastRange 在并发调用下会被别的调用覆盖）
static thread_local int sTlsLastRange[6] = { 0, 0, 0, 0, 0, 0 };
struct sNarrowPhaseTimer
{
	std::chrono::steady_clock::time_point mStart;
	uint64 mV0, mS0, mP0, mSc0;
	sNarrowPhaseTimer() : mStart(std::chrono::steady_clock::now())
	{
		VoxelShape::sStatNarrowPhaseCalls.fetch_add(1, std::memory_order_relaxed);
		mV0 = VoxelShape::sStatVisitedCells.load(std::memory_order_relaxed);
		mS0 = VoxelShape::sStatSolidCells.load(std::memory_order_relaxed);
		mP0 = VoxelShape::sStatProbes.load(std::memory_order_relaxed);
		mSc0 = VoxelShape::sStatScanCells.load(std::memory_order_relaxed);
	}
	~sNarrowPhaseTimer()
	{
		uint64 ns = uint64(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - mStart).count());
		VoxelShape::sStatNarrowPhaseNs.fetch_add(ns, std::memory_order_relaxed);
		// 单次调用最大耗时：判断总耗时是"1 次独大"还是"每次调用的固定开销均摊"
		uint64 cur = VoxelShape::sStatNarrowPhaseMaxNs.load(std::memory_order_relaxed);
		while (ns > cur)
		{
			if (VoxelShape::sStatNarrowPhaseMaxNs.compare_exchange_weak(cur, ns, std::memory_order_relaxed))
			{
				// CAS 赢了：本调用就是新的最大 ⇒ 快照本调用的计数差值
				std::lock_guard<std::mutex> lock(VoxelShape::sNarrowPhaseMaxMutex);
				VoxelShape::sNarrowPhaseMaxVisited = VoxelShape::sStatVisitedCells.load(std::memory_order_relaxed) - mV0;
				VoxelShape::sNarrowPhaseMaxSolid = VoxelShape::sStatSolidCells.load(std::memory_order_relaxed) - mS0;
				VoxelShape::sNarrowPhaseMaxProbes = VoxelShape::sStatProbes.load(std::memory_order_relaxed) - mP0;
				VoxelShape::sNarrowPhaseMaxProbeNs = sCallProbeLoopNs;
				VoxelShape::sNarrowPhaseMaxWalkNs = sCallWalkNs > sCallProbeLoopNs? sCallWalkNs - sCallProbeLoopNs : 0;
				VoxelShape::sNarrowPhaseMaxSubTests = sCallSubTests;
				VoxelShape::sNarrowPhaseMaxSubEmpty = sCallSubEmpty;
				VoxelShape::sNarrowPhaseMaxScan = VoxelShape::sStatScanCells.load(std::memory_order_relaxed) - mSc0;
				for (int k = 0; k < 6; ++k)
					VoxelShape::sNarrowPhaseMaxRange[k] = sTlsLastRange[k];
				break;
			}
			// CAS 输了：cur 被更新成更大的值，循环条件重判
		}
	}
};

void VoxelShape::GetSolidRange(int outMin[3], int outMax[3]) const
{
	EnsureDerived();
	for (int k = 0; k < 3; ++k)
	{
		outMin[k] = mSolidMin[k];
		outMax[k] = mSolidMax[k];
	}
}


/// 体素 vs 体素：接触法线（= 深度方向、接触点推出方向）取自哪里。详见 VoxelShape.h 里的说明书。
///
///   0 = 整网格 SAT 全局轴：所有格对共用一根轴，深嵌时接触方向一致不互相抵消；
///       但曲面/斜面接触给出与接触面无关的方向（甜甜圈落圆锥被顶飞、球放环心自转）。
///   1 = 逐对格心连线（纯球语义，内切球）：n = normalize(格心2 - 格心1)，深度 = 内切球半径和 − |Δc|
///       （与球筛同半径）；整格埋住（2x2x2 八格全实心）时改用全局轴。边际成本最低，但平地轴对齐
///       接触会把错位格对的假穿透当真（见 VoxelShape.h 的详细说明）。
///   2 = **局部 2x2x2 邻域分类**（默认）：把遍历侧那一格映射到候选侧格空间（在那里它是
///       轴对齐的 AABB），用它圈出候选侧的 2x2x2 邻域，数这 8 格里**候选侧**的实心格，
///       按排布定法线：1 格 -> 球语义；2 格同轴 -> 胶囊；4 格同面 -> 面法线（指向空格侧）；
///       7 格 -> 推向唯一空格；6 格 -> 推向两个空格连成的槽；8 格 -> 完全埋住，用全局轴；
///       其余（3/5 格、对角 2 格、两条平行棱…）-> 贪心拆成"面 > 棱 > 单格"加权合并。
///       为什么需要它：整网格 SAT 的 half 由 size 决定、**不含空气**，大网格里"90% 是空气"时全局轴
///       退化成"离哪个盒面最近"（8 m 地形块 vs 0.5 m 碎块 ⇒ 碎块贴边时法线变纯水平、左右横跳）。
///       2x2x2 只看得见眼前的 8 格 ⇒ 方向由真实重叠轮廓决定；且全部在格空间里算，不需要投影。
///
/// ★ 默认 = **2（局部 2x2x2 邻域分类）**。历史：2026-10-29 曾默认 1（球语义）—— 球语义两侧对称，
///   一堆碎片堆在一起时接触点/法线两边一致，碎片堆能正常进入休眠；局部 2x2x2 邻域分类以候选侧
///   为基准、不保证两侧对称，碎片堆叠时接触一致性不如球语义（两种语义的取舍见 VoxelShape.h）。
int VoxelShape::sNormalMode = 2;
int VoxelShape::sBlockMaskCache = 1;


/// 见 VoxelShape.h 里 sLastGridOverlap 的说明。只用于诊断，不参与碰撞逻辑。
/// atomic：窄相位是多线程的，多个 worker 会同时写这两项（普通 float 数组那是数据竞争）。
atomic<float> VoxelShape::sLastGridOverlap[2] { { 0.0f }, { 0.0f } };

// 轴符号修正的统计（诊断用，经 joltc 的 JPH_VoxelShape_GetAxisFixStats 暴露给 C#；
// sFixAxisSign 已按用户约定删除，这些计数器保留恒为 0，API 不动）：
// [0]=调用次数  [1]=未干预（派生无效 / 轴不平行 / 交集为空 / 两侧都实心或对称）
// [2]=按探测结果定了符号  [3]=其中与传入轴符号相反（真的翻转）的次数
atomic<uint32_t> VoxelShape::sAxisFixStats[4] { { 0 }, { 0 }, { 0 }, { 0 } };
atomic<uint32_t> VoxelShape::sDiagBranch[8] { { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 }, { 0 } };
bool VoxelShape::sDiagBranchEnabled = false;


//////////////////////////////////////////////////////////////////////////////////////////
// 构造
//////////////////////////////////////////////////////////////////////////////////////////

ShapeSettings::ShapeResult VoxelShapeSettings::Create() const
{
	if (mCachedResult.IsEmpty())
		Ref<Shape> shape = new VoxelShape(*this, mCachedResult);
	return mCachedResult;
}

VoxelShape::VoxelShape(const VoxelShapeSettings &inSettings, ShapeResult &outResult) :
	Shape(EShapeType::User1, EShapeSubType::User1, inSettings, outResult),
	mSizeX(inSettings.mSizeX),
	mSizeY(inSettings.mSizeY),
	mSizeZ(inSettings.mSizeZ),
	mVoxelSize(inSettings.mVoxelSize),
	mOccupancy(inSettings.mOccupancy),
	mTexelsX(sGetTexelCount(inSettings.mSizeX)),
	mTexelsY(sGetTexelCount(inSettings.mSizeY)),
	mTexelPlane(sGetTexelCount(inSettings.mSizeX) * sGetTexelCount(inSettings.mSizeY)),
	mDensity(inSettings.mDensity),
	mMaterial(inSettings.mMaterial)
{
	// 网格尺寸
	if (mSizeX <= 0 || mSizeY <= 0 || mSizeZ <= 0)
	{
		outResult.SetError("Invalid voxel grid size");
		return;
	}

	// 体素边长
	if (mVoxelSize <= 0.0f)
	{
		outResult.SetError("Invalid voxel size");
		return;
	}

	// 占用位图（形状不拥有它，但必须保证在形状存活期间一直有效）
	if (mOccupancy == nullptr)
	{
		outResult.SetError("Voxel data is null");
		return;
	}

	// 密度
	if (mDensity < 0.0f)
	{
		outResult.SetError("Invalid density");
		return;
	}

	// 算出半长、实心格数量、SubShapeID 位数和质量属性
	Recalculate();

	outResult.Set(this);
}

VoxelShape::VoxelShape(int inSizeX, int inSizeY, int inSizeZ, float inVoxelSize, const uint64 *inOccupancy, float inDensity) :
	Shape(EShapeType::User1, EShapeSubType::User1),
	mSizeX(inSizeX),
	mSizeY(inSizeY),
	mSizeZ(inSizeZ),
	mVoxelSize(inVoxelSize),
	mOccupancy(inOccupancy),
	mTexelsX(sGetTexelCount(inSizeX)),
	mTexelsY(sGetTexelCount(inSizeY)),
	mTexelPlane(sGetTexelCount(inSizeX) * sGetTexelCount(inSizeY)),
	mDensity(inDensity)
{
	// 这条路径没有 ShapeResult，所以参数合法性只在内部做防御性处理（见 Recalculate 和 IsSolid）
	Recalculate();
}

//////////////////////////////////////////////////////////////////////////////////////////
// 内部辅助
//////////////////////////////////////////////////////////////////////////////////////////

uint VoxelShape::sGetBitsForSize(int inSize)
{
	// 至少要 1 位（SubShapeIDCreator::PushID 会断言 inValue < (1 << inBits)）
	uint bits = 1;
	while (bits < 32 && (uint(1) << bits) < uint(inSize))
		++bits;
	return bits;
}

const BoxShape *VoxelShape::sGetUnitBox()
{
	// 进程级单例，故意不释放（体素形状的生命周期基本等同于进程）。
	//
	// 关键点：凸半径必须显式给 0。BoxShape 默认会带 cDefaultConvexRadius（0.05 m）的凸半径，
	// 那样每一格都会被"膨出"一个圆角，相邻格子互相穿插，细缝/薄墙会被糊死，所以这里必须传 0。
	static const BoxShape *sUnitBox = []() {
		BoxShape *box = new BoxShape(Vec3::sReplicate(0.5f), 0.0f);
		box->AddRef();
		return box;
	}();
	return sUnitBox;
}

const SphereShape *VoxelShape::sGetUnitSphere()
{
	// 进程级单例，故意不释放（与 sGetUnitBox 同一套约定）。
	//
	// sUseSphereContacts 开启时，体素 vs 其它形状的窄相位用这个单位球代替单位盒：
	// 半径 0.5（内切球），每格按格缩放乘上去。球 vs 任意形状是**点接触**（1 个点，无面），
	// 一个 10 x 10 的落地面的接触点数从 100 格 x 4 点 = 400 降到 100 —— 而且球 vs 凸/球的
	// 查询本身也比盒 vs 凸的 GJK/EPA 便宜。
	//
	// 角点是已知代价：格子的棱角不再参与碰撞（球缩进去了），贴墙走/棱上站立的手感会更"圆"。
	// CastRay / shape cast 不用球（射线没有体积，用球会有洞），仍然走盒子路径。
	static const SphereShape *sUnitSphere = []() {
		SphereShape *sphere = new SphereShape(0.5f);
		sphere->AddRef();
		return sphere;
	}();
	return sUnitSphere;
}

const CapsuleShape *VoxelShape::sGetUnitCapsule(int inRunLength)
{
	JPH_ASSERT(inRunLength >= 2 && inRunLength <= cMaxRunLength);

	// 进程级胶囊池（与 sGetUnitBox / sGetUnitSphere 同一套约定）。半径固定 0.5（半格），长度全靠圆柱
	// 半高 (L - 1) / 2 承担：相邻格的球心间距 1、半径 0.5，串起来的 L 个球正好相切，中间的空隙由圆柱
	// 填满 —— 乘上均匀的格缩放之后就是一根正好覆盖 L 个格子的胶囊。
	//
	// ★ 一次建好、之后只读：窄相位在多线程上跑（PhysicsSystem 的 CollideBodiesJob），所以这里不能有
	//   惰性写入。函数内 static 的初始化本身是线程安全的。
	struct Pool
	{
		const CapsuleShape *mCapsules[cMaxRunLength + 1] = { };

		Pool()
		{
			for (int i = 2; i <= cMaxRunLength; ++i)
			{
				CapsuleShape *capsule = new CapsuleShape(0.5f * float(i - 1), 0.5f);
				capsule->AddRef();
				mCapsules[i] = capsule;
			}
		}
	};
	static const Pool sPool;
	return sPool.mCapsules[inRunLength];
}

bool VoxelShape::sGetCellRange(const AABox &inLocalBox, int &outMinX, int &outMinY, int &outMinZ, int &outMaxX, int &outMaxY, int &outMaxZ) const
{
	if (mOccupancy == nullptr || mVoxelSize <= 0.0f || mSizeX <= 0 || mSizeY <= 0 || mSizeZ <= 0)
		return false;

	// 空盒子（例如两个形状的包围盒根本不相交）
	if (inLocalBox.mMin.GetX() > inLocalBox.mMax.GetX()
		|| inLocalBox.mMin.GetY() > inLocalBox.mMax.GetY()
		|| inLocalBox.mMin.GetZ() > inLocalBox.mMax.GetZ())
		return false;

	const float bmin[3] = { inLocalBox.mMin.GetX(), inLocalBox.mMin.GetY(), inLocalBox.mMin.GetZ() };
	const float bmax[3] = { inLocalBox.mMax.GetX(), inLocalBox.mMax.GetY(), inLocalBox.mMax.GetZ() };
	const float extent[3] = { mHalfExtent.GetX(), mHalfExtent.GetY(), mHalfExtent.GetZ() };
	const int size[3] = { mSizeX, mSizeY, mSizeZ };
	// 没做过惰性统计时这两个 getter 返回整张网格（不触发扫描，只是少剪一点）
	const int solid_min[3] = { GetSolidMin(0), GetSolidMin(1), GetSolidMin(2) };
	const int solid_max[3] = { GetSolidMax(0), GetSolidMax(1), GetSolidMax(2) };

	int lo[3], hi[3];
	for (int axis = 0; axis < 3; ++axis)
	{
		// 格子 c 在该轴上占据 [c * s - H, (c + 1) * s - H]，
		// 于是"坐标 v 属于哪一格"就是 (v + H) / s 往下取整。
		float inv_voxel_size = 1.0f / mVoxelSize;
		float fmin = (bmin[axis] + extent[axis]) * inv_voxel_size;
		float fmax = (bmax[axis] + extent[axis]) * inv_voxel_size;

		// 先把范围夹住，避免极端值/NaN 走进 int 转换的未定义行为
		fmin = Clamp(fmin, -1.0e7f, 1.0e7f);
		fmax = Clamp(fmax, -1.0e7f, 1.0e7f);

		// 覆盖到的格子下标区间（闭区间）：下界往上取整、上界往下取整，
		// 这样"正好贴在一起"的两格只有真正重叠时才会同时落在区间里。
		int l = int(std::floor(fmin + cCellEpsilon));
		int h = int(std::ceil(fmax - cCellEpsilon)) - 1;

		// 夹进网格，再和"实心格包围范围"求交（外围那些空格子不用遍历）
		l = max(l, max(0, solid_min[axis]));
		h = min(h, min(size[axis] - 1, solid_max[axis]));
		if (l > h)
			return false;

		lo[axis] = l;
		hi[axis] = h;
	}

	outMinX = lo[0];
	outMinY = lo[1];
	outMinZ = lo[2];
	outMaxX = hi[0];
	outMaxY = hi[1];
	outMaxZ = hi[2];
	return true;
}

template <bool tOutsideIn, bool tSurfaceOnly, class F>
void VoxelShape::sVisitSolidVoxelsInBox(const AABox &inLocalBox, Vec3Arg inVoxelScale, Mat44Arg inVoxelTransform, const F &inVisitor) const
{
	int min_x, min_y, min_z, max_x, max_y, max_z;
	if (!sGetCellRange(inLocalBox, min_x, min_y, min_z, max_x, max_y, max_z))
		return;

	// 施加在"格子单位盒子"上的缩放：形状缩放 * 体素边长
	Vec3 cell_scale = inVoxelScale * mVoxelSize;

	// 把格子单位盒子从体素局部空间搬到目标空间：目标变换 * 形状缩放 * 平移到格心
	Mat44 voxel_to_target = inVoxelTransform * Mat44::sScale(inVoxelScale);

	// 格心目标坐标 = cell_origin + (ix + 0.5) * step[0] + (iy + 0.5) * step[1] + (iz + 0.5) * step[2]
	//
	// ★ 原来每格写的是 voxel_to_target * GetVoxelCenter(...) —— 一次 4x4 矩阵乘。而 GetVoxelCenter
	//   是仿射的（(i+0.5)*格长 - halfExtent），所以整条链可以拆成 基点 + 三个步进向量，
	//   每格只剩 3 次乘加。这是最内层循环，实测遍历开销的主要部分就在这里。
	//   （0.5 折进基点：O + Σ(i+0.5)*S = (O + 0.5*(S0+S1+S2)) + Σ i*S）
	// ★ 步进向量是**方向**，必须用 Multiply3x3（线性部分）：用点变换会把平移量加进每一步。
	const Vec3 cell_step[3] =
	{
		voxel_to_target.Multiply3x3(Vec3(mVoxelSize, 0, 0)),
		voxel_to_target.Multiply3x3(Vec3(0, mVoxelSize, 0)),
		voxel_to_target.Multiply3x3(Vec3(0, 0, mVoxelSize)),
	};
	const Vec3 cell_origin = voxel_to_target * (-mHalfExtent) + 0.5f * (cell_step[0] + cell_step[1] + cell_step[2]);

	// 下标序列：tOutsideIn 时从两端往中间（0, n-1, 1, n-2, ...），否则就是顺序
	auto index = [](int inLo, int inHi, int inI)
	{
		if constexpr (tOutsideIn)
			return (inI & 1) != 0? inHi - (inI >> 1) : inLo + (inI >> 1);
		else
			return inLo + inI;
	};

	// 访问一格：算好格子变换再交给访问者（等价于 voxel_to_target * Mat44::sTranslation(格心)，
	// 但格心直接用基点 + 三个步进向量算出来，省掉每格一次 4x4 矩阵乘 —— 这里是最内层循环，很敏感）
	// ★★ 交给访问者的变换必须是**纯旋转 + 平移**：格的缩放是作为"形状缩放"（cell_scale）单独传下去的，
	//   绝不能再出现在变换的 3x3 里。窄相位把 3x3 当正交旋转来用（CollideConvexVsConvex 里
	//   InversedRotationTranslation() 取的是它的转置），里面塞了缩放会让整条查询算错 ——
	//   非均匀格缩放下实测就是"一格都碰不上、0 命中"。
	//   （格心坐标仍然用带缩放的 voxel_to_target 去算，缩放只从 3x3 里去掉。）
	auto visit_cell = [&](int inX, int inY, int inZ)
	{
		if (sHotStatsEnabled.load(std::memory_order_relaxed))
			sStatScanCells.fetch_add(1, std::memory_order_relaxed);		// 诊断：扫到的实心格（含被表面过滤滤掉的）
		// 表面枚举（tSurfaceOnly）：跳过内部格（6 邻全实心）。与 run_cell 里 mode 2 的"任一侧
		// 内部就跳"语义一致 —— 内部格在该过滤下本来就产不出任何接触，这里提前跳掉，省掉
		// "枚举整球体积"的访问成本。缓存未建时先惰性建一次（EnsureDerived 幂等）；建不出
		// （occupancy 为空）就不过滤，绝不静默丢弃全部格子。
		if constexpr (tSurfaceOnly)
		{
			if (mSurfaceBits.empty())
				EnsureDerived();
			if (!mSurfaceBits.empty() && !IsVoxelSurface(inX, inY, inZ))
				return;
		}

		if (sHotStatsEnabled.load(std::memory_order_relaxed))
			sStatVisitedCells.fetch_add(1, std::memory_order_relaxed);		// 调试统计：遍历过的格（含空格）
		Mat44 cell_transform = inVoxelTransform;
		cell_transform.SetTranslation(cell_origin + float(inX) * cell_step[0] + float(inY) * cell_step[1] + float(inZ) * cell_step[2]);
		inVisitor(inX, inY, inZ, cell_transform, cell_scale);
	};

	// 三层推进：8x8x8 的 chunk（整块空就跳过 512 格）-> 4x4x4 的子块（空就跳过 64 格）-> 格子。
	// 每层都把"该块的遍历区间"和请求区间求交，边界块只走它真正参与的那部分。
	int min_chunk_x = min_x >> cChunkShift, max_chunk_x = max_x >> cChunkShift;
	int min_chunk_y = min_y >> cChunkShift, max_chunk_y = max_y >> cChunkShift;
	int min_chunk_z = min_z >> cChunkShift, max_chunk_z = max_z >> cChunkShift;

	for (int cz = 0; cz <= max_chunk_z - min_chunk_z; ++cz)
	{
		int chunk_z = index(min_chunk_z, max_chunk_z, cz);
		int chunk_z_base = chunk_z << cChunkShift;

		for (int cy = 0; cy <= max_chunk_y - min_chunk_y; ++cy)
		{
			int chunk_y = index(min_chunk_y, max_chunk_y, cy);
			int chunk_y_base = chunk_y << cChunkShift;

			for (int cx = 0; cx <= max_chunk_x - min_chunk_x; ++cx)
			{
				int chunk_x = index(min_chunk_x, max_chunk_x, cx);
				int chunk_x_base = chunk_x << cChunkShift;

				for (int sz = 0; sz < cNumSubChunksPerAxis; ++sz)
				{
					int sub_z = index(0, cNumSubChunksPerAxis - 1, sz);
					int z_base = chunk_z_base + (sub_z << cSubChunkShift);
					int z0 = max(min_z, z_base), z1 = min(max_z, z_base + cSubChunkSize - 1);

					for (int sy = 0; sy < cNumSubChunksPerAxis; ++sy)
					{
						int sub_y = index(0, cNumSubChunksPerAxis - 1, sy);
						int y_base = chunk_y_base + (sub_y << cSubChunkShift);
						int y0 = max(min_y, y_base), y1 = min(max_y, y_base + cSubChunkSize - 1);

						for (int sx = 0; sx < cNumSubChunksPerAxis; ++sx)
						{
							int sub_x = index(0, cNumSubChunksPerAxis - 1, sx);
							int x_base = chunk_x_base + (sub_x << cSubChunkShift);
							int x0 = max(min_x, x_base), x1 = min(max_x, x_base + cSubChunkSize - 1);

							// 这个 4x4x4 子块里一格实心都没有 -> 一次 64 位比较跳过 64 格。
							// 字直接取自占用位图（越界坐标返回 0，与早先掩码的 nullptr 语义等价），
							// 所以就地编辑之后不需要 Recalculate 也立刻生效。
							// ★ 表面枚举（tSurfaceOnly）：改读**表面位图**——子块字为 0 = 64 格里没有一格
							//   表面（全内部或全空），一次比较跳过整个子块。深陷球实测每查询要扫 16 万个
							//   内部格、只有 ~9 千表面格，"扫内部格再被表面过滤扔掉"是枚举的主成本；
							//   直接按表面位图枚举把这部分降为"只在含表面格的子块里逐格走"。位图未建时
							//   惰性建一次（EnsureDerived 幂等）；建不出退回占用位图（语义 = 不过滤）。
							// ★ 子块基点可能落在"幽灵区"：网格某轴尺寸不是 8 的倍数时（mod 8 ∈ 1..4），
							//   最后一个 chunk 的第二个子块（base+4）超出 mSize。GetWord 有越界钳制
							//   （返回 0），mSurfaceBits 直接下标没有 —— Debug 下 JPH_ASSERT(inIdx < mSize)
							//   直接断死，Release 下越界读到邻层/相邻内存（实测 32×32×12：z_base=12 →
							//   texel 层 3 ≥ texelsZ=3，index 196 ≥ 192）。
							const bool sub_oob = x_base >= mSizeX || y_base >= mSizeY || z_base >= mSizeZ;
							uint64 sub;
							if constexpr (tSurfaceOnly)
							{
								if (mSurfaceBits.empty())
									EnsureDerived();
								sub = (sub_oob || mSurfaceBits.empty())? uint64(0)
									: mSurfaceBits[GetTexelIndex(x_base, y_base, z_base)];
							}
							else
								sub = GetWord(x_base, y_base, z_base);
							++sCallSubTests;
							if (x0 > x1 || y0 > y1 || z0 > z1 || sub == 0)
							{
								if (sub == 0)
									++sCallSubEmpty;
								continue;
							}

							for (int iz = 0; iz <= z1 - z0; ++iz)
							{
								int z = index(z0, z1, iz);
								for (int iy = 0; iy <= y1 - y0; ++iy)
								{
									int y = index(y0, y1, iy);

									// ★ 一行里的 4 个 x 落在**同一个字**里（x 占最低 2 位），所以只有行首算一次
									//   位偏移，逐格就只剩"移位 + 测试"：既不用再判越界（x0..x1 已被
									//   sGetCellRange clamp 进网格），也不用每格访存一次。
									int row_shift = ((y & cSubChunkMask) << cSubChunkShift) | ((z & cSubChunkMask) << (2 * cSubChunkShift));
									// 表面枚举：整行 nibble 为 0（4 格全非表面）再省掉逐格测试
									if constexpr (tSurfaceOnly)
										if (((sub >> row_shift) & 0xFULL) == 0)
											continue;
									for (int ix = 0; ix <= x1 - x0; ++ix)
									{
										int x = index(x0, x1, ix);
										if ((sub >> (row_shift + (x & cSubChunkMask))) & 1)
											visit_cell(x, y, z);
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

template <class F>
void VoxelShape::sVisitSolidVoxelRunsInBox(const AABox &inLocalBox, Vec3Arg inVoxelScale, Mat44Arg inVoxelTransform, const F &inVisitor) const
{
	int min_x, min_y, min_z, max_x, max_y, max_z;
	if (!sGetCellRange(inLocalBox, min_x, min_y, min_z, max_x, max_y, max_z))
		return;

	// 与 sVisitSolidVoxelsInBox 完全相同的一套"格心 = 基点 + Σ 下标 * 步进"（0.5 已折进基点），
	// 这里只是把最内层的 x 从"逐格回调"改成"扫游程"。
	Mat44 voxel_to_target = inVoxelTransform * Mat44::sScale(inVoxelScale);
	const Vec3 cell_step[3] =
	{
		voxel_to_target.Multiply3x3(Vec3(mVoxelSize, 0, 0)),
		voxel_to_target.Multiply3x3(Vec3(0, mVoxelSize, 0)),
		voxel_to_target.Multiply3x3(Vec3(0, 0, mVoxelSize)),
	};
	const Vec3 cell_origin = voxel_to_target * (-mHalfExtent) + 0.5f * (cell_step[0] + cell_step[1] + cell_step[2]);

	// 游程方向在整条查询里是常量，提到循环外只归一化一次
	const Vec3 run_axis = cell_step[0].Normalized();

	// 输出一段游程（[inXStart, inXEnd] 闭区间）
	auto emit_segment = [&](int inXStart, int inXEnd, int inY, int inZ)
	{
		// 游程中心 = 首尾格心的中点 = 基点 + 中间下标 * 步进
		float mid = 0.5f * float(inXStart + inXEnd);
		Vec3 center = cell_origin + mid * cell_step[0] + float(inY) * cell_step[1] + float(inZ) * cell_step[2];

		// 首格的变换（平移量 = 首格格心）。只有单格游程会用到它。
		// ★ 旋转部分取自**不带缩放**的 inVoxelTransform，原因见 sVisitSolidVoxelsInBox 里的说明。
		Mat44 first_cell = inVoxelTransform;
		first_cell.SetTranslation(cell_origin + float(inXStart) * cell_step[0] + float(inY) * cell_step[1] + float(inZ) * cell_step[2]);

		inVisitor(inXStart, inXEnd - inXStart + 1, inY, inZ, first_cell, center, run_axis);
	};

	// 超过胶囊池上限的游程拆成多段（尾段可能只剩 1 格，调用方会用内切球兜底）
	auto emit_run = [&](int inXStart, int inXEnd, int inY, int inZ)
	{
		while (inXEnd - inXStart + 1 > cMaxRunLength)
		{
			emit_segment(inXStart, inXStart + cMaxRunLength - 1, inY, inZ);
			inXStart += cMaxRunLength;
		}
		emit_segment(inXStart, inXEnd, inY, inZ);
	};

	int min_chunk_x = min_x >> cChunkShift, max_chunk_x = max_x >> cChunkShift;
	int min_chunk_y = min_y >> cChunkShift, max_chunk_y = max_y >> cChunkShift;
	int min_chunk_z = min_z >> cChunkShift, max_chunk_z = max_z >> cChunkShift;

	// y / z 走分块（整块空就跳过 64 行），x 在行内扫 —— 一行是连续内存，不需要再分块。
	for (int cz = min_chunk_z; cz <= max_chunk_z; ++cz)
	{
		int cz_base = cz << cChunkShift;
		int z0 = max(min_z, cz_base), z1 = min(max_z, cz_base + cChunkSize - 1);

		for (int cy = min_chunk_y; cy <= max_chunk_y; ++cy)
		{
			int cy_base = cy << cChunkShift;
			int y0 = max(min_y, cy_base), y1 = min(max_y, cy_base + cChunkSize - 1);

			for (int z = z0; z <= z1; ++z)
				for (int y = y0; y <= y1; ++y)
				{
					// 这一行的位偏移（行内不变）：移位后低 4 位就是该行 4 个 x 位置的实心位
					int shift = ((z & cSubChunkMask) << 4) | ((y & cSubChunkMask) << cSubChunkShift);

					int run_start = -1, run_end = -1;

					// cx / sx 两层循环让 x 单调递增，游程可以跨 chunk 与子块延续
					for (int cx = min_chunk_x; cx <= max_chunk_x; ++cx)
					{
						int cx_base = cx << cChunkShift;
						int xa = max(min_x, cx_base), xb = min(max_x, cx_base + cChunkSize - 1);

						for (int sx = 0; sx < cNumSubChunksPerAxis; ++sx)
						{
							int sub_base = cx_base + (sx << cSubChunkShift);
							int x0 = max(xa, sub_base), x1 = min(xb, sub_base + cSubChunkSize - 1);
							if (x0 > x1)
								continue;												// 该子块不在遍历范围内

							// 这 4 格共用一个字（= 它们所属的那个 4x4x4 子块）；越界坐标返回 0 = 整组为空
							uint64 word = GetWord(sub_base, y, z);

							// 这个 4 格组全空 -> 收尾当前游程，并一次跳过 4 格
							if (((word >> shift) & 0xF) == 0)
							{
								if (run_start >= 0)
								{
									emit_run(run_start, run_end, y, z);
									run_start = -1;
								}
								continue;
							}

							for (int x = x0; x <= x1; ++x)
							{
								// 读的是**实时**的占用位（不是派生掩码），就地编辑后不需要 Recalculate 也立刻生效
								if (((word >> (shift + (x & cSubChunkMask))) & 1) == 0)
								{
									if (run_start >= 0)
									{
										emit_run(run_start, run_end, y, z);
										run_start = -1;
									}
								}
								else if (run_start >= 0)
								{
									run_end = x;											// 与当前游程相接 -> 延长
								}
								else
								{
									run_start = run_end = x;								// 刚从空开始 -> 起新游程
								}
							}
						}
					}

					if (run_start >= 0)
						emit_run(run_start, run_end, y, z);
				}
		}
	}
}

bool VoxelShape::FindClosestVoxelFace(int inX, int inY, int inZ, Vec3Arg inLocalPosition, Vec3 &outNormal) const
{
	// 6 个面的外法线
	static const Vec3 cNormals[6] =
	{
		Vec3( 1,  0,  0), Vec3(-1,  0,  0),
		Vec3( 0,  1,  0), Vec3( 0, -1,  0),
		Vec3( 0,  0,  1), Vec3( 0,  0, -1)
	};

	// 从格心沿法线走半个格子就到那个面
	Vec3 center = GetVoxelCenter(inX, inY, inZ);
	float half = 0.5f * mVoxelSize;

	float best_distance = FLT_MAX;
	bool found = false;
	for (int i = 0; i < 6; ++i)
	{
		// 只考虑"邻居是空的"那些面：内部面即使更近也推不出去
		int dx = int(cNormals[i].GetX());
		int dy = int(cNormals[i].GetY());
		int dz = int(cNormals[i].GetZ());
		if (!IsVoxelFaceExposed(inX, inY, inZ, dx, dy, dz))
			continue;

		// 点到该面所在平面的（有符号）距离，取最小的那个面
		float distance = (center + cNormals[i] * half - inLocalPosition).Dot(cNormals[i]);
		if (distance < best_distance)
		{
			best_distance = distance;
			outNormal = cNormals[i];
			found = true;
		}
	}

	return found;
}

namespace
{
	/// 4x4x4 子块字里的位序：bit = x | (y << 2) | (z << 4)（x 占最低 2 位，与 VoxelShape::sGetSubChunkBit 一致）。
	/// 下表把"某一轴上取某个值 0..3"的那些位全部挑出来，配合 popcount 就能从**一个字**直接得到
	/// 每条轴上"每个取值各有多少格"，不需要把 64 个位逐格展开。
	struct WordAxisMasks
	{
		uint64	mMask[3][4] = { };

		constexpr WordAxisMasks()
		{
			for (int z = 0; z < 4; ++z)
				for (int y = 0; y < 4; ++y)
					for (int x = 0; x < 4; ++x)
					{
						uint64 bit = uint64(1) << (x | (y << 2) | (z << 4));
						mMask[0][x] |= bit;
						mMask[1][y] |= bit;
						mMask[2][z] |= bit;
					}
		}
	};

	constexpr WordAxisMasks sWordAxisMasks;

	/// 64 位 popcount
	inline int sPopCount(uint64 inValue)
	{
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
		return int(__popcnt64(inValue));
#elif defined(__GNUC__) || defined(__clang__)
		return __builtin_popcountll(inValue);
#else
		// 兜底：SWAR popcount
		inValue = inValue - ((inValue >> 1) & 0x5555555555555555ull);
		inValue = (inValue & 0x3333333333333333ull) + ((inValue >> 2) & 0x3333333333333333ull);
		inValue = (inValue + (inValue >> 4)) & 0x0f0f0f0f0f0f0f0full;
		return int((inValue * 0x0101010101010101ull) >> 56);
#endif
	}
}

void VoxelShape::EnsureDerived() const
{
	// ★ 快路径：已建好 ⇒ 无锁 bool 读，热路径零开销（99.9% 的调用走这里）。
	if (mDerivedValid)
		return;

	// ★ 首次构建要持锁：窄相位在多个 job 线程上跑，没预热的形状（如分块体的子形状）可能被
	//   两个线程同时构建 —— 对空 Array 同时 resize ⇒ 双重分配/指针撕裂 ⇒ 读到悬空缓冲
	//   （0xC0000005，2026-10-28 开碎块后崩溃的根因）。锁内双重检查：等锁期间别人建完了就走。
	lock_guard<mutex> lock(mDerivedMutex);
	if (mDerivedValid)
		return;

	// 诊断：快速路径（已建好）不计；真正重建的次数与耗时全记（这是"每次调用固定开销"的头号嫌疑）。
	const std::chrono::steady_clock::time_point ed_t0 = std::chrono::steady_clock::now();
	mDerivedValid = true;

	// ---- 质量属性常量 ----
	// 每一格是一个均匀的小立方体：质量 = 体积 * 密度，绕自身质心的转动惯量 = m * s^2 / 6（三个轴一样）。
	// 用平行轴定理把它搬到网格中心；因为格子都是轴对齐的，最终惯量张量是对角的。
	float cell_volume = Cubed(mVoxelSize);
	float cell_mass = cell_volume * mDensity;
	float cell_inertia = cell_mass * Square(mVoxelSize) / 6.0f;

	uint num_solid = 0;
	int min_cell[3] = { mSizeX, mSizeY, mSizeZ };
	int max_cell[3] = { -1, -1, -1 };
	double sum_c[3] = { 0.0, 0.0, 0.0 };		// Σ 格心坐标（GetMassCenter 的分子）
	double sum_r_sq[3] = { 0.0, 0.0, 0.0 };		// Σ 格心坐标²，三个轴各一个（相对网格中心）

	const int texels_x = mTexelsX;
	const int texels_y = mTexelsY;
	const int texels_z = sGetTexelCount(mSizeZ);
	const int num_texels = texels_x * texels_y * texels_z;

	// ---- 8^3 chunk 摘要（与下面的逐字扫描同一趟，边际成本 ≈ 0）----
	// 窄相位用它做两件事：遍历侧按"实心格数"而不是"盒格数"选（80^3 球贴着 128^3 地形时，
	// 球侧在重叠区里只有几百个实心格、地形侧有一万多，按盒格数两者相当、会选错）；以及
	// "候选侧没有实心 chunk / 双方都是全内部 chunk"的整块剪枝。
	const int chunk_dims[3] =
	{
		(mSizeX + cChunkMask) >> cChunkShift,
		(mSizeY + cChunkMask) >> cChunkShift,
		(mSizeZ + cChunkMask) >> cChunkShift,
	};
	const int chunk_plane = chunk_dims[0] * chunk_dims[1];
	const size_t chunk_total = size_t(chunk_dims[0]) * chunk_dims[1] * chunk_dims[2];
	mChunkSolid.resize(chunk_total);
	mChunkLayerSolid.resize(chunk_total * cChunkSize);
	mChunkTight.resize(chunk_total);
	for (size_t i = 0; i < chunk_total; ++i)
	{
		mChunkSolid[i] = 0;
		mChunkLayerSolid[i * cChunkSize] = 0; mChunkLayerSolid[i * cChunkSize + 1] = 0;
		mChunkLayerSolid[i * cChunkSize + 2] = 0; mChunkLayerSolid[i * cChunkSize + 3] = 0;
		mChunkLayerSolid[i * cChunkSize + 4] = 0; mChunkLayerSolid[i * cChunkSize + 5] = 0;
		mChunkLayerSolid[i * cChunkSize + 6] = 0; mChunkLayerSolid[i * cChunkSize + 7] = 0;
		// 紧界初值：min = 15（任何 0..7 的实心格都会把它拉下来），max = 0
		mChunkTight[i] = uint64(0xF) | (uint64(0xF) << 6) | (uint64(0xF) << 12);
	}

	if (mOccupancy != nullptr && num_texels > 0 && mSizeX > 0 && mSizeY > 0 && mSizeZ > 0)
	{
		// 尺寸不是 4 的整数倍时，最后一行/一层的字里有"越界位"，必须掩掉（位图按 4 对齐分配）
		const bool has_edge = (mSizeX & cSubChunkMask) != 0 || (mSizeY & cSubChunkMask) != 0 || (mSizeZ & cSubChunkMask) != 0;
		const float half[3] = { 0.5f * mSizeX * mVoxelSize, 0.5f * mSizeY * mVoxelSize, 0.5f * mSizeZ * mVoxelSize };
		const bool need_inertia = cell_mass > 0.0f;

		int tx = 0, ty = 0, tz = 0;
		const uint64 *word = mOccupancy;
		for (int i = 0; i < num_texels; ++i, ++word)
		{
			uint64 w = *word;

			if (has_edge)
			{
				int base[3] = { tx << cSubChunkShift, ty << cSubChunkShift, tz << cSubChunkShift };
				if (base[0] + cSubChunkSize > mSizeX || base[1] + cSubChunkSize > mSizeY || base[2] + cSubChunkSize > mSizeZ)
				{
					uint64 valid = 0;
					for (int lz = 0; lz < cSubChunkSize; ++lz)
						for (int ly = 0; ly < cSubChunkSize; ++ly)
							for (int lx = 0; lx < cSubChunkSize; ++lx)
								if (base[0] + lx < mSizeX && base[1] + ly < mSizeY && base[2] + lz < mSizeZ)
									valid |= uint64(1) << (lx | (ly << 2) | (lz << 4));
					w &= valid;
				}
			}

			if (w != 0)
			{
				const int pc = sPopCount(w);
				num_solid += uint(pc);
				const size_t tight_idx = size_t(tx >> 1) + size_t(ty >> 1) * chunk_dims[0] + size_t(tz >> 1) * chunk_plane;
				uint64 tight = mChunkTight[tight_idx];
				mChunkSolid[(tx >> 1) + (ty >> 1) * chunk_dims[0] + (tz >> 1) * chunk_plane] += uint16(pc);

				for (int axis = 0; axis < 3; ++axis)
				{
					int base = (axis == 0? tx : axis == 1? ty : tz) << cSubChunkShift;
					for (int v = 0; v < 4; ++v)
					{
						int count = sPopCount(w & sWordAxisMasks.mMask[axis][v]);
						if (count == 0)
							continue;

						int cell = base + v;
						if (cell < min_cell[axis]) min_cell[axis] = cell;
						if (cell > max_cell[axis]) max_cell[axis] = cell;

						// ★ 实心格紧界（与 mChunkSolid 同一趟，边际成本 ≈ 0）：cell & 7 = chunk 内下标
						{
							const int b = cell & cChunkMask;
							const int smin = axis == 0? 0 : axis == 1? 6 : 12;
							int cmin = int(tight >> smin) & 7;
							int cmax = int(tight >> (smin + 3)) & 7;
							if (b < cmin) tight = (tight & ~(7ull << smin)) | (uint64(b) << smin);
							if (b > cmax) tight = (tight & ~(7ull << (smin + 3))) | (uint64(b) << (smin + 3));
						}

						// Y 分层计数（轴 1 = y）：窄相位的遍历侧选择按它算 range 内实心格数
						// ★ chunk 下标必须与 mChunkSolid 同一套 (x,y,z) 排布 —— 2026-10-24 修正：
						//   旧版这里用 (x,z,y)，增量编辑（sApplyVoxelEdit）用 (x,y,z)，编辑过的地形计数写错位。
						if (axis == 1)
							mChunkLayerSolid[(size_t(tx >> 1) + size_t(ty >> 1) * chunk_dims[0]
								+ size_t(tz >> 1) * chunk_plane) * cChunkSize + (cell & cChunkMask)] += uint8(count);

						// 格心坐标（相对网格中心）= (cell + 0.5) * 体素边长 - 半长
						// ★ 顺手把 Σ 坐标累加出来（GetMassCenter 用）：这里的 count 已经是"该轴上
						//   取这个值的格数"，所以一次乘加就等价于逐格累加 64 次 —— 边际成本为 0。
						double coord = (double(cell) + 0.5) * double(mVoxelSize) - double(half[axis]);
						sum_c[axis] += double(count) * coord;

						if (need_inertia)
							sum_r_sq[axis] += double(count) * coord * coord;
					}
				}
				mChunkTight[tight_idx] = tight;
			}

			// 推进 texel 坐标（x 最快，与 GetTexelIndex 的排版一致）
			if (++tx == texels_x)
			{
				tx = 0;
				if (++ty == texels_y)
				{
					ty = 0;
					++tz;
				}
			}
		}
	}

	mNumSolidVoxels = num_solid;
	mSolidMin[0] = min_cell[0]; mSolidMin[1] = min_cell[1]; mSolidMin[2] = min_cell[2];
	mSolidMax[0] = max_cell[0]; mSolidMax[1] = max_cell[1]; mSolidMax[2] = max_cell[2];
	mMassCenterSum[0] = sum_c[0]; mMassCenterSum[1] = sum_c[1]; mMassCenterSum[2] = sum_c[2];

	// ---- 质量属性 ----
	// 惯量对角元（绕网格中心）：Ix = Σ[cell_inertia + m*(ry² + rz²)] = N*cell_inertia + m*(Σry² + Σrz²)
	// ★ 用 double 存：增量编辑（UpdateSolid）要在它上面做差值加减，float 在"大数减小数"上会产生
	//   可观的消减误差（实测只剩 1 格时相对误差 100%+），double 压到 1e-10 级。
	if (num_solid > 0 && cell_mass > 0.0f)
	{
		double inertia_cells = double(num_solid) * double(cell_inertia);
		mInertiaDiag[0] = inertia_cells + double(cell_mass) * (sum_r_sq[1] + sum_r_sq[2]);
		mInertiaDiag[1] = inertia_cells + double(cell_mass) * (sum_r_sq[0] + sum_r_sq[2]);
		mInertiaDiag[2] = inertia_cells + double(cell_mass) * (sum_r_sq[0] + sum_r_sq[1]);
	}
	else
		mInertiaDiag[0] = mInertiaDiag[1] = mInertiaDiag[2] = 0.0;

	mMassProperties = MassProperties();
	mMassProperties.mMass = float(num_solid) * cell_mass;
	mMassProperties.mInertia = Mat44::sIdentity(); // 没有实心格时给一个单位惯量，避免刚体属性变成 NaN
	if (num_solid > 0 && cell_mass > 0.0f)
	{
		mMassProperties.mInertia = Mat44::sZero();
		mMassProperties.mInertia.SetDiagonal3(Vec3(float(mInertiaDiag[0]), float(mInertiaDiag[1]), float(mInertiaDiag[2])));
		mMassProperties.mInertia.SetColumn4(3, Vec4(0, 0, 0, 1));
	}

	// ---- 表面格位图（mode 2 表面枚举的过滤源；之后由 sApplyVoxelEdit 增量维护）----
	BuildSurfaceCache();

	// 诊断：真正重建（快速路径不计）的次数与耗时 —— 判"派生缓存被反复失效/新建形状首次碰撞"偷了多少时间
	sStatEnsureDerivedCalls.fetch_add(1, std::memory_order_relaxed);
	sStatEnsureDerivedNs.fetch_add(uint64(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - ed_t0).count()), std::memory_order_relaxed);
}

void VoxelShape::BuildSurfaceCache() const
{
	if (mOccupancy == nullptr || mSizeX <= 0 || mSizeY <= 0 || mSizeZ <= 0)
	{
		mSurfaceBits.clear();
		mChunkSurface.clear();
		mChunkLayerSurface.clear();
		return;
	}

	mSurfaceBits.resize(size_t(mTexelsX) * mTexelsY * sGetTexelCount(mSizeZ));
	memset(mSurfaceBits.data(), 0, mSurfaceBits.size() * sizeof(uint64));

	const int cd[3] =
	{
		(mSizeX + cChunkMask) >> cChunkShift,
		(mSizeY + cChunkMask) >> cChunkShift,
		(mSizeZ + cChunkMask) >> cChunkShift,
	};
	const int chunk_plane = cd[0] * cd[1];
	mChunkSurface.resize(size_t(cd[0]) * cd[1] * cd[2]);
	mChunkLayerSurface.resize(mChunkSurface.size() * cChunkSize);
	memset(mChunkSurface.data(), 0, mChunkSurface.size() * sizeof(uint16));
	memset(mChunkLayerSurface.data(), 0, mChunkLayerSurface.size() * sizeof(uint8));

	for (int z = 0; z < mSizeZ; ++z)
		for (int y = 0; y < mSizeY; ++y)
		{
			// 行首算一次位基址，逐格只做移位测试（与 sVisitSolidVoxelsInBox 的行内优化同款）
			int row_shift = ((y & cSubChunkMask) << cSubChunkShift) | ((z & cSubChunkMask) << (2 * cSubChunkShift));
			for (int x = 0; x < mSizeX; ++x)
			{
				if (!IsSolid(x, y, z))
					continue;
				bool surface = !IsSolid(x + 1, y, z) || !IsSolid(x - 1, y, z)
					|| !IsSolid(x, y + 1, z) || !IsSolid(x, y - 1, z)
					|| !IsSolid(x, y, z + 1) || !IsSolid(x, y, z - 1);
				if (!surface)
					continue;
				mSurfaceBits[GetTexelIndex(x, y, z)] |= uint64(1) << sGetSubChunkBit(x, y, z);
				const size_t cidx = size_t(x >> cChunkShift) + size_t(y >> cChunkShift) * cd[0] + size_t(z >> cChunkShift) * chunk_plane;
				++mChunkSurface[cidx];
				++mChunkLayerSurface[cidx * cChunkSize + (y & cChunkMask)];
			}
			(void)row_shift;
		}
}

Vec3 VoxelShape::GetMassCenter() const
{
	// ★ 累加量在 EnsureDerived 里顺带算好（同一趟按 4x4x4 字的扫描，见那里的说明）：
	//   旧实现是这里另起一遍三重循环逐格 IsSolid + GetVoxelCenter —— 128³ 的块要问 200 万次
	//   "你这格实心吗"，而其中 99% 是空格。
	if (mOccupancy == nullptr)
		return Vec3::sZero();

	EnsureDerived();
	if (mNumSolidVoxels == 0)
		return Vec3::sZero();

	float inv = 1.0f / float(mNumSolidVoxels);
	return Vec3(float(mMassCenterSum[0]) * inv, float(mMassCenterSum[1]) * inv, float(mMassCenterSum[2]) * inv);
}

int VoxelShape::CountSolidCellsInRange(const int *inRange) const
{
	// 摘要无效（没统计过 / occupancy 没扫过）时让调用方回退盒格数
	if (!mDerivedValid || mChunkSolid.empty())
		return -1;

	const int cd[3] =
	{
		(mSizeX + cChunkMask) >> cChunkShift,
		(mSizeY + cChunkMask) >> cChunkShift,
		(mSizeZ + cChunkMask) >> cChunkShift,
	};
	const int cmin[3] =
	{
		max(0, inRange[0]) >> cChunkShift,
		max(0, inRange[1]) >> cChunkShift,
		max(0, inRange[2]) >> cChunkShift,
	};
	const int cmax[3] =
	{
		min(cd[0] - 1, inRange[3] >> cChunkShift),
		min(cd[1] - 1, inRange[4] >> cChunkShift),
		min(cd[2] - 1, inRange[5] >> cChunkShift),
	};

	// x/z 按 chunk 累加；y 把每个 chunk 里与 range 相交的层逐层加（y 是球贴地场景里
	// 唯一被"切薄"的轴，必须按 1 格精度，否则球侧的计数会被整块 chunk 的实心数撑爆）
	int total = 0;
	for (int cz = cmin[2]; cz <= cmax[2]; ++cz)
		for (int cx = cmin[0]; cx <= cmax[0]; ++cx)
		{
			// chunk 下标 (x,y,z) 排布，与 mChunkSolid / 增量编辑一致（2026-10-24 修正：旧版 (x,z,y)）
			const size_t col = size_t(cx) + size_t(cz) * cd[0] * cd[1];
			for (int cy = cmin[1]; cy <= cmax[1]; ++cy)
			{
				const size_t chunk = col + size_t(cy) * cd[0];
				const int ylo = max(inRange[1], cy << cChunkShift);
				const int yhi = min(inRange[4], (cy << cChunkShift) + cChunkMask);
				for (int y = ylo; y <= yhi; ++y)
					total += mChunkLayerSolid[chunk * cChunkSize + (y & cChunkMask)];
			}
		}
	return total;
}

int VoxelShape::DebugVisitSurfaceBox(const AABox &inBox) const
{
	int n = 0;
	sVisitSolidVoxelsInBox<false, true>(inBox, Vec3::sOne(), Mat44::sIdentity(), [&](int, int, int, const Mat44 &, Vec3) { ++n; });
	return n;
}

int VoxelShape::CountSurfaceCellsInRange(const int *inRange) const
{
	// 与 CountSolidCellsInRange 同一套摘要口径，只是换读表面格摘要（sInteriorSkipPairs == 2 时
	// 遍历侧按"表面格数"选 —— 球贴地形时地形侧表面只有几千、球侧上万，按实心数会选错边）。
	if (!mDerivedValid || mChunkSurface.empty())
		return -1;

	const int cd[3] =
	{
		(mSizeX + cChunkMask) >> cChunkShift,
		(mSizeY + cChunkMask) >> cChunkShift,
		(mSizeZ + cChunkMask) >> cChunkShift,
	};
	const int cmin[3] =
	{
		max(0, inRange[0]) >> cChunkShift,
		max(0, inRange[1]) >> cChunkShift,
		max(0, inRange[2]) >> cChunkShift,
	};
	const int cmax[3] =
	{
		min(cd[0] - 1, inRange[3] >> cChunkShift),
		min(cd[1] - 1, inRange[4] >> cChunkShift),
		min(cd[2] - 1, inRange[5] >> cChunkShift),
	};

	int total = 0;
	for (int cz = cmin[2]; cz <= cmax[2]; ++cz)
		for (int cx = cmin[0]; cx <= cmax[0]; ++cx)
		{
			// chunk 下标 (x,y,z) 排布，与 BuildSurfaceCache / 增量编辑一致（2026-10-24 修正：旧版 (x,z,y)）
			const size_t col = size_t(cx) + size_t(cz) * cd[0] * cd[1];
			for (int cy = cmin[1]; cy <= cmax[1]; ++cy)
			{
				const size_t chunk = col + size_t(cy) * cd[0];
				const int ylo = max(inRange[1], cy << cChunkShift);
				const int yhi = min(inRange[4], (cy << cChunkShift) + cChunkMask);
				for (int y = ylo; y <= yhi; ++y)
					total += mChunkLayerSurface[chunk * cChunkSize + (y & cChunkMask)];
			}
		}
	return total;
}

bool VoxelShape::IsChunkAllInterior(int inCx, int inCy, int inCz) const
{
	if (!mDerivedValid || mChunkSolid.empty())
		return false;

	const int cd[3] =
	{
		(mSizeX + cChunkMask) >> cChunkShift,
		(mSizeY + cChunkMask) >> cChunkShift,
		(mSizeZ + cChunkMask) >> cChunkShift,
	};
	if (inCx < 0 || inCy < 0 || inCz < 0 || inCx >= cd[0] || inCy >= cd[1] || inCz >= cd[2])
		return false;

	// 自身全实心（512 格），且 6 个邻 chunk 也全实心；越界邻 chunk 视为空（与 IsSolid 一致）
	if (mChunkSolid[size_t(inCx) + size_t(inCy) * cd[0] + size_t(inCz) * cd[0] * cd[1]] != cChunkVolume)
		return false;

	static constexpr int sOffsets[6][3] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
	for (const int (&off)[3] : sOffsets)
	{
		const int nx = inCx + off[0], ny = inCy + off[1], nz = inCz + off[2];
		if (nx < 0 || ny < 0 || nz < 0 || nx >= cd[0] || ny >= cd[1] || nz >= cd[2])
			return false;
		if (mChunkSolid[size_t(nx) + size_t(ny) * cd[0] + size_t(nz) * cd[0] * cd[1]] != cChunkVolume)
			return false;
	}
	return true;
}

void VoxelShape::Recalculate() const
{
	// 半长（未缩放局部空间）
	mHalfExtent = Vec3(0.5f * mSizeX * mVoxelSize, 0.5f * mSizeY * mVoxelSize, 0.5f * mSizeZ * mVoxelSize);

	// 把一个体素坐标塞进 SubShapeID 需要多少位
	mBitsX = sGetBitsForSize(mSizeX);
	mBitsY = sGetBitsForSize(mSizeY);
	mBitsZ = sGetBitsForSize(mSizeZ);

	// ★ 这里**不扫描体素**。实心格数 / 实心包围范围 / 质量属性 / 质心累加全部改成惰性统计
	//   （EnsureDerived），只有真正被问到才算：动态刚体建体时取质量、外部问 GetNumSolidVoxels、
	//   或增量编辑要一个基线。地形区块是静态体，质量属性不参与求解 ⇒ 建区块时这里只写几个常量，
	//   成本与体积无关。
	mDerivedValid = false;
	mNumSolidVoxels = 0;
	mSolidMin[0] = 0; mSolidMin[1] = 0; mSolidMin[2] = 0;
	mSolidMax[0] = -1; mSolidMax[1] = -1; mSolidMax[2] = -1;
	mMassProperties = MassProperties();
	mInertiaDiag[0] = mInertiaDiag[1] = mInertiaDiag[2] = 0.0;
	mMassCenterSum[0] = mMassCenterSum[1] = mMassCenterSum[2] = 0.0;

	// 让缓存的调试几何失效（几何里已经把 mVoxelSize 烘进去了，体素数据一改就得重建）。
	// 版本号取**全局唯一**的下一个值（不是本形状自增）：它同时是"形状身份"和"数据世代"。
	mVersion = sNextVoxelDataVersion.fetch_add(1, memory_order_relaxed);
#ifdef JPH_DEBUG_RENDERER
	mGeometry = nullptr;
	mGeometryVersion = uint(-1);
#endif // JPH_DEBUG_RENDERER
}

bool VoxelShape::sApplyVoxelEdit(int inX, int inY, int inZ, bool inSolid) const
{
	// 越界坐标忽略（与 SetSolid() 一致）
	if (mOccupancy == nullptr || inX < 0 || inX >= mSizeX || inY < 0 || inY >= mSizeY || inZ < 0 || inZ >= mSizeZ)
		return false;

	// 增量维护要有正确的基线：没统计过就先统计一遍（之后一直是增量、O(1)）
	EnsureDerived();

	// 占用位图里这一格的位。写入即生效（查询读的就是它），与邻居无关 ⇒ 一次 64 位读改写。
	uint64 &word = const_cast<uint64 *>(mOccupancy)[GetTexelIndex(inX, inY, inZ)];
	uint64 bit = uint64(1) << sGetSubChunkBit(inX, inY, inZ);
	bool was_solid = (word & bit) != 0;
	if (was_solid == inSolid)
		return false;											// 实心性没变：几何 / 缓存 / 数据版本都不受影响

	// 写位（等价于 SetSolid()，只是把"要不要换数据版本"的决定权交给调用方）
	if (inSolid)
		word |= bit;
	else
		word &= ~bit;

	// ---- 计数 ----
	mNumSolidVoxels += inSolid? 1 : uint(-1);

	// ---- 8^3 chunk 摘要：O(1) 增量（摘要在 EnsureDerived 里建好，编辑只动这一格所在 chunk 的计数）----
	{
		const int cdx = (mSizeX + cChunkMask) >> cChunkShift;
		const int cdy = (mSizeY + cChunkMask) >> cChunkShift;
		const size_t chunk_idx = size_t(inX >> cChunkShift)
			+ size_t(inY >> cChunkShift) * cdx
			+ size_t(inZ >> cChunkShift) * (cdx * cdy);
		const int delta = inSolid? 1 : -1;
		int total = int(mChunkSolid[chunk_idx]) + delta;
		mChunkSolid[chunk_idx] = uint16(total < 0? 0 : total);
		int layer = int(mChunkLayerSolid[chunk_idx * cChunkSize + (inY & cChunkMask)]) + delta;
		mChunkLayerSolid[chunk_idx * cChunkSize + (inY & cChunkMask)] = uint8(layer < 0? 0 : layer);

		// 紧界：本 chunk 有格变化，min/max 可能收缩（删掉角落的格）⇒ 直接重扫这 512 格
		//（O(chunk) 与形状大小无关；mChunkTight 与 mChunkSolid 同在 EnsureDerived 建好基线）
		sRescanChunkTight(chunk_idx);
	}

	// ---- 表面位图增量：这一格 + 6 个邻居的"表面性"都可能变（邻居的暴露面取决于这一格）----
	//（EnsureDerived 建好基线后一直是 O(1)；mSurfaceBits 为空说明派生统计从未建过，无需维护）
	if (!mSurfaceBits.empty())
	{
		static constexpr int sNbOff[7][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
		const int cdx = (mSizeX + cChunkMask) >> cChunkShift;
		const int cdy = (mSizeY + cChunkMask) >> cChunkShift;
		const int chunk_plane = cdx * cdy;
		for (const int (&off)[3] : sNbOff)
		{
			const int x = inX + off[0], y = inY + off[1], z = inZ + off[2];
			if (x < 0 || x >= mSizeX || y < 0 || y >= mSizeY || z < 0 || z >= mSizeZ)
				continue;

			const bool surface = IsSolid(x, y, z)
				&& (!IsSolid(x + 1, y, z) || !IsSolid(x - 1, y, z)
					|| !IsSolid(x, y + 1, z) || !IsSolid(x, y - 1, z)
					|| !IsSolid(x, y, z + 1) || !IsSolid(x, y, z - 1));
			const size_t wi = GetTexelIndex(x, y, z);
			const uint64 sb = uint64(1) << sGetSubChunkBit(x, y, z);
			if (((mSurfaceBits[wi] & sb) != 0) == surface)
				continue;										// 表面性没变

			if (surface)
				mSurfaceBits[wi] |= sb;
			else
				mSurfaceBits[wi] &= ~sb;

			const size_t cidx = size_t(x >> cChunkShift) + size_t(y >> cChunkShift) * cdx + size_t(z >> cChunkShift) * chunk_plane;
			const int sdelta = surface? 1 : -1;
			int stotal = int(mChunkSurface[cidx]) + sdelta;
			mChunkSurface[cidx] = uint16(stotal < 0? 0 : stotal);
			const size_t li = cidx * cChunkSize + (y & cChunkMask);
			int ltotal = int(mChunkLayerSurface[li]) + sdelta;
			mChunkLayerSurface[li] = uint8(ltotal < 0? 0 : ltotal);
		}
	}

	// ---- 实心包围范围：**只扩不缩**（删除造成的收缩留给下一次 Recalculate()）----
	if (inSolid)
	{
		if (mNumSolidVoxels == 1)
		{
			// 之前一个实心格都没有 => 直接以这一格为界
			mSolidMin[0] = inX; mSolidMin[1] = inY; mSolidMin[2] = inZ;
			mSolidMax[0] = inX; mSolidMax[1] = inY; mSolidMax[2] = inZ;
		}
		else
		{
			if (inX < mSolidMin[0]) mSolidMin[0] = inX;
			if (inY < mSolidMin[1]) mSolidMin[1] = inY;
			if (inZ < mSolidMin[2]) mSolidMin[2] = inZ;
			if (inX > mSolidMax[0]) mSolidMax[0] = inX;
			if (inY > mSolidMax[1]) mSolidMax[1] = inY;
			if (inZ > mSolidMax[2]) mSolidMax[2] = inZ;
		}
	}
	else if (mNumSolidVoxels == 0)
	{
		// 删空了：与 Recalculate() 在没有实心格时的结果对齐（min = size，max = -1）
		mSolidMin[0] = mSizeX; mSolidMin[1] = mSizeY; mSolidMin[2] = mSizeZ;
		mSolidMax[0] = -1; mSolidMax[1] = -1; mSolidMax[2] = -1;
	}

	// ---- 质量属性：质量按计数重算（避免长期累加的漂移），惯量与质心累加按差值加减 ----
	float cell_volume = Cubed(mVoxelSize);
	float cell_mass = cell_volume * mDensity;
	float cell_inertia = cell_mass * Square(mVoxelSize) / 6.0f;

	mMassProperties.mMass = float(mNumSolidVoxels) * cell_mass;

	Vec3 r = GetVoxelCenter(inX, inY, inZ);
	double sign = inSolid? 1.0 : -1.0;

	// 质心累加（GetMassCenter 的分子）：与密度无关，所以放在 cell_mass 的判断之外 ——
	// 密度为 0 时质量/惯量都无意义，但"实心格在哪里"仍然要维护，否则 GetMassCenter 会给出旧值。
	mMassCenterSum[0] += sign * double(r.GetX());
	mMassCenterSum[1] += sign * double(r.GetY());
	mMassCenterSum[2] += sign * double(r.GetZ());

	if (cell_mass > 0.0f)
	{
		double r_sq = double(r.LengthSq());
		mInertiaDiag[0] += sign * (double(cell_inertia) + double(cell_mass) * (r_sq - double(r.GetX()) * double(r.GetX())));
		mInertiaDiag[1] += sign * (double(cell_inertia) + double(cell_mass) * (r_sq - double(r.GetY()) * double(r.GetY())));
		mInertiaDiag[2] += sign * (double(cell_inertia) + double(cell_mass) * (r_sq - double(r.GetZ()) * double(r.GetZ())));
	}

	if (mNumSolidVoxels > 0 && cell_mass > 0.0f)
	{
		mMassProperties.mInertia = Mat44::sZero();
		mMassProperties.mInertia.SetDiagonal3(Vec3(float(mInertiaDiag[0]), float(mInertiaDiag[1]), float(mInertiaDiag[2])));
		mMassProperties.mInertia.SetColumn4(3, Vec4(0, 0, 0, 1));
	}
	else
	{
		// 没有实心格（或密度为 0）时给一个单位惯量，与 Recalculate() 一致，避免刚体属性 NaN
		mMassProperties.mInertia = Mat44::sIdentity();
	}

	return true;
}

void VoxelShape::UpdateSolid(int inX, int inY, int inZ, bool inSolid) const
{
	if (sApplyVoxelEdit(inX, inY, inZ, inSolid))
	{
		// 实心性变了 => 数据世代变了：边缘格缓存里的旧条目（含这一格与它 6 个邻居的）会失配并自动重算，
		// 缓存的调试几何也要重建。
		mVersion = sNextVoxelDataVersion.fetch_add(1, memory_order_relaxed);
#ifdef JPH_DEBUG_RENDERER
		mGeometry = nullptr;
		mGeometryVersion = uint(-1);
#endif // JPH_DEBUG_RENDERER
	}
}

void VoxelShape::UpdateSolids(const SolidEdit *inEdits, int inCount) const
{
	if (inEdits == nullptr || inCount <= 0)
		return;

	bool changed = false;
	for (int i = 0; i < inCount; ++i)
	{
		const SolidEdit &edit = inEdits[i];
		changed |= sApplyVoxelEdit(edit.mX, edit.mY, edit.mZ, edit.mSolid);
	}

	// 整批只换一次数据版本（破坏一片体素时的常态）
	if (changed)
	{
		mVersion = sNextVoxelDataVersion.fetch_add(1, memory_order_relaxed);
#ifdef JPH_DEBUG_RENDERER
		mGeometry = nullptr;
		mGeometryVersion = uint(-1);
#endif // JPH_DEBUG_RENDERER
	}
}

void VoxelShape::sRescanChunkTight(size_t inChunkIdx) const
{
	const int cdx = (mSizeX + cChunkMask) >> cChunkShift;
	const int cdy = (mSizeY + cChunkMask) >> cChunkShift;
	uint64 tight = uint64(0xF) | (uint64(0xF) << 6) | (uint64(0xF) << 12);
	const int bx = int(inChunkIdx % cdx) << cChunkShift;
	const int by = int((inChunkIdx / cdx) % cdy) << cChunkShift;
	const int bz = int(inChunkIdx / (cdx * cdy)) << cChunkShift;
	for (int lz = 0; lz < cChunkSize; ++lz)
		for (int ly = 0; ly < cChunkSize; ++ly)
			for (int lx = 0; lx < cChunkSize; ++lx)
			{
				const int x = bx + lx, y = by + ly, z = bz + lz;
				if (x >= mSizeX || y >= mSizeY || z >= mSizeZ || !IsSolid(x, y, z))
					continue;
				const int v[3] = { lx, ly, lz };
				for (int axis = 0; axis < 3; ++axis)
				{
					const int smin = axis == 0? 0 : axis == 1? 6 : 12;
					if (v[axis] < int(tight >> smin & 7))
						tight = (tight & ~(7ull << smin)) | (uint64(v[axis]) << smin);
					if (v[axis] > int(tight >> (smin + 3) & 7))
						tight = (tight & ~(7ull << (smin + 3))) | (uint64(v[axis]) << (smin + 3));
				}
			}
	mChunkTight[inChunkIdx] = tight;
}

void VoxelShape::sShrinkChunkTight(size_t inChunkIdx, const uint64 *inRemovedWords) const
{
	// ★ 增量紧界收缩：调用时被挖位已从占用位图清掉（ApplyCarveBitmap 第 1 步写回之后）。
	//   挖格只会让紧界收缩：某轴 min 变的唯一可能是「x == 旧 min 的所有实心格都被挖掉」，
	//   所以被挖格 AABB 没碰到旧界面的轴直接不动；碰到的轴从旧界面开始逐层找第一个
	//   仍含实心的层（IsSolid 读的就是新位图）。全空时与 sRescanChunkTight 的空 chunk
	//   结果一致（min=7 > max=1 哨兵）。
	if (inRemovedWords == nullptr)
	{
		sRescanChunkTight(inChunkIdx);
		return;
	}

	// 被挖格的 chunk 局部 AABB：每 word 64 位里按轴掩码提取（8 word × 12 次与运算，
	// 比逐位扫 512 次便宜得多）。位序 b = (x&3)|((y&3)<<2)|((z&3)<<4)。
	static constexpr uint64 sAxisMask[3][4] = {
		{ 0x1111111111111111ull, 0x2222222222222222ull, 0x4444444444444444ull, 0x8888888888888888ull },	// x
		{ 0x000F000F000F000Full, 0x00F000F000F000F0ull, 0x0F000F000F000F00ull, 0xF000F000F000F000ull },	// y
		{ 0x000000000000FFFFull, 0x00000000FFFF0000ull, 0x0000FFFF00000000ull, 0xFFFF000000000000ull },	// z
	};
	int dmin[3] = { 8, 8, 8 }, dmax[3] = { -1, -1, -1 };
	for (int i = 0; i < 8; ++i)
	{
		const uint64 rem = inRemovedWords[i];
		if (rem == 0)
			continue;
		const int base[3] = { (i & 1) * 4, ((i >> 1) & 1) * 4, (i >> 2) * 4 };
		for (int axis = 0; axis < 3; ++axis)
			for (int t = 0; t < 4; ++t)
				if (rem & sAxisMask[axis][t])
				{
					const int g = base[axis] + t;
					if (g < dmin[axis]) dmin[axis] = g;
					if (g > dmax[axis]) dmax[axis] = g;
				}
	}

	const int cdx = (mSizeX + cChunkMask) >> cChunkShift;
	const int cdy = (mSizeY + cChunkMask) >> cChunkShift;
	const int bx = int(inChunkIdx % cdx) << cChunkShift;
	const int by = int((inChunkIdx / cdx) % cdy) << cChunkShift;
	const int bz = int(inChunkIdx / (cdx * cdy)) << cChunkShift;

	// 层判定：axis 上第 t 层（另两轴 0..7 全查）。
	// ★★★ 2026-11-07：改成**按 4³ 字**判定，块内 64 格零逐格遍历。
	//   8³ chunk = 2x2x2 个 4³ 字；第 t 层落在 t>>2 那一半的字里，字内偏移 t&3。
	//   每字跟该轴的层掩码做一次 AND，非零即该层仍有实心 ⇒ 每层 4 次读字，
	//   旧版是 8x8 = 64 次 IsSolid；整块挖空时旧版还要连探 8 层，这里照样 4 次。
	//   8³ chunk 完全在网格内（调用方保证 16³ 块整体落界内）⇒ 字下标不越界。
	auto layer_has_solid = [this, bx, by, bz](int inAxis, int inT)
	{
		const int half = inT >> 2;
		const uint64 plane_mask = sAxisMask[inAxis][inT & 3];
		for (int a = 0; a < 2; ++a)
			for (int b = 0; b < 2; ++b)
			{
				int lx = half, ly = a, lz = b;
				if (inAxis == 1)		{ lx = a; ly = half; lz = b; }
				else if (inAxis == 2)	{ lx = a; ly = b; lz = half; }
				const uint64 word = mOccupancy[GetTexelIndex(bx + lx * 4, by + ly * 4, bz + lz * 4)];
				if ((word & plane_mask) != 0)
					return true;
			}
		return false;
	};

	uint64 tight = mChunkTight[inChunkIdx];
	int tmin[3] = { int(tight & 7), int((tight >> 6) & 7), int((tight >> 12) & 7) };
	int tmax[3] = { int((tight >> 3) & 7), int((tight >> 9) & 7), int((tight >> 15) & 7) };

	bool empty = false;
	for (int axis = 0; axis < 3 && !empty; ++axis)
	{
		if (dmin[axis] > tmin[axis] && dmax[axis] < tmax[axis])
			continue;										// 两个界面都没被挖到 ⇒ 该轴紧界必不变
		if (dmin[axis] <= tmin[axis])						// min 面被挖到：从旧 min 往里找
		{
			int found = -1;
			for (int t = tmin[axis]; t <= tmax[axis] && found < 0; ++t)
				if (layer_has_solid(axis, t)) found = t;
			if (found < 0) { empty = true; break; }			// 一整轴都无实心 ⇒ chunk 全空
			tmin[axis] = found;
		}
		if (dmax[axis] >= tmax[axis])						// max 面被挖到：从旧 max 往里找
		{
			int found = -1;
			for (int t = tmax[axis]; t >= tmin[axis] && found < 0; --t)
				if (layer_has_solid(axis, t)) found = t;
			if (found < 0) { empty = true; break; }
			tmax[axis] = found;
		}
	}

	if (empty)
		tight = uint64(0xF) | (uint64(0xF) << 6) | (uint64(0xF) << 12);		// 与全扫版空 chunk 一致
	else
		tight = uint64(tmin[0]) | (uint64(tmax[0]) << 3)
			| (uint64(tmin[1]) << 6) | (uint64(tmax[1]) << 9)
			| (uint64(tmin[2]) << 12) | (uint64(tmax[2]) << 15);
	mChunkTight[inChunkIdx] = tight;
}

bool VoxelShape::sRefreshSolidBounds() const
{
	// ★ 实心格包围范围 = 所有非空 8³ chunk 紧界的并集。mChunkTight 在编辑路径上一直
	//   增量维护（sRescanChunkTight / sShrinkChunkTight），合成只是 O(chunk 数) 次整型比较。
	// ★★★ 为什么必须随挖洞收缩：sSolidAABBWorld / sGetCellRange / 4³ 记账全用这个范围
	//   圈「双方实心盒的交集」。旧版只扩不缩 ⇒ 挖空的形状仍用整网格大小的实心盒与地形
	//   相交，窄相位每个物理步把整片空区走一遍（现场 Jolt/Step 恒定 10~15ms/步）。
	EnsureDerived();

	int mn[3] = { INT_MAX, INT_MAX, INT_MAX };
	int mx[3] = { INT_MIN, INT_MIN, INT_MIN };
	const int cdx = (mSizeX + cChunkMask) >> cChunkShift;
	const int cdy = (mSizeY + cChunkMask) >> cChunkShift;
	const int cdz = (mSizeZ + cChunkMask) >> cChunkShift;
	for (int ci = 0; ci < cdx * cdy * cdz; ++ci)
	{
		const uint64 tight = mChunkTight[ci];
		const int tmin[3] = { int(tight & 7), int((tight >> 6) & 7), int((tight >> 12) & 7) };
		const int tmax[3] = { int((tight >> 3) & 7), int((tight >> 9) & 7), int((tight >> 15) & 7) };
		if (tmin[0] > tmax[0] || tmin[1] > tmax[1] || tmin[2] > tmax[2])
			continue;										// 空 chunk（哨兵 min>max）
		const int bx = (ci % cdx) << cChunkShift;
		const int by = ((ci / cdx) % cdy) << cChunkShift;
		const int bz = (ci / (cdx * cdy)) << cChunkShift;
		const int base[3] = { bx, by, bz };
		for (int axis = 0; axis < 3; ++axis)
		{
			if (base[axis] + tmin[axis] < mn[axis]) mn[axis] = base[axis] + tmin[axis];
			if (base[axis] + tmax[axis] > mx[axis]) mx[axis] = base[axis] + tmax[axis];
		}
	}
	if (mn[0] > mx[0] || mn[1] > mx[1] || mn[2] > mx[2])
	{
		// 全空：与 Recalculate() 相同的哨兵，sSolidAABBWorld 据此返回空盒
		mSolidMin[0] = mSizeX; mSolidMin[1] = mSizeY; mSolidMin[2] = mSizeZ;
		mSolidMax[0] = -1; mSolidMax[1] = -1; mSolidMax[2] = -1;
		return false;
	}
	for (int axis = 0; axis < 3; ++axis)
	{
		mSolidMin[axis] = mn[axis];
		mSolidMax[axis] = mx[axis];
	}
	return true;
}

bool VoxelShape::SetCenterOfMassToSolidBounds() const
{
	// ★ 质心 = 实心包围盒中心（用户口径：不做逐体素累加）。
	//   结果与"逐体素累加得到的质量质心"不同（这是有意的：包围盒中心随挖洞方向移动，
	//   支点偏向剩余几何的那一侧，翻倒行为更直观）。
	if (!sRefreshSolidBounds())
		return false;										// 全空：质心保持原值
	const int mn[3] = { mSolidMin[0], mSolidMin[1], mSolidMin[2] };
	const int mx[3] = { mSolidMax[0], mSolidMax[1], mSolidMax[2] };

	// 格块 [mn, mx] 的几何中心 → 相对形状原点（= 网格中心）：
	//   格 x 的中心 = (x + 0.5)·vs − halfExtent，块中心 = ((mn + mx + 1)·0.5)·vs − halfExtent
	SetCenterOfMass(Vec3(
		(float(mn[0]) + float(mx[0]) + 1.0f) * (0.5f * mVoxelSize) - mHalfExtent.GetX(),
		(float(mn[1]) + float(mx[1]) + 1.0f) * (0.5f * mVoxelSize) - mHalfExtent.GetY(),
		(float(mn[2]) + float(mx[2]) + 1.0f) * (0.5f * mVoxelSize) - mHalfExtent.GetZ()));
	++mVersion;												// 质心参与 bounds/变换 ⇒ 版本换代
	return true;
}

uint VoxelShape::ApplyCarveBitmap(int inBaseX, int inBaseY, int inBaseZ, const uint64 *inBits) const
{
	// chunk 必须整块落在网格内：分块体的网格 = chunk 整数倍 ⇒ 恒满足；单网格 / 越界交给调用方逐格回退
	if (mOccupancy == nullptr || inBits == nullptr
		|| inBaseX < 0 || inBaseY < 0 || inBaseZ < 0
		|| inBaseX + 16 > mSizeX || inBaseY + 16 > mSizeY || inBaseZ + 16 > mSizeZ)
		return uint(-1);

	EnsureDerived();

	const int cdx = (mSizeX + cChunkMask) >> cChunkShift;
	const int cdy = (mSizeY + cChunkMask) >> cChunkShift;
	const int chunk_plane = cdx * cdy;

	uint64 *occ = const_cast<uint64 *>(mOccupancy);

	// 实际清掉的位（字下标 = chunk 局部 0..63，与入参同布局）：表面位图 / 质量属性两步按它逐格处理
	uint64 removed_bits[64];
	uint64 dirty_words = 0;
	uint total_removed = 0;

	// ---- 1. 逐字 AND-NOT + 计数（实心总数 / 8³ chunk 摘要 / 逐层）----
	for (int wz = 0; wz < 4; ++wz)
		for (int wy = 0; wy < 4; ++wy)
			for (int wx = 0; wx < 4; ++wx)
			{
				const int w = wx + wy * 4 + wz * 16;
				removed_bits[w] = 0;
				const uint64 want = inBits[w];
				if (want == 0)
					continue;
				const int x = inBaseX + wx * 4, y = inBaseY + wy * 4, z = inBaseZ + wz * 4;
				uint64 &word = occ[GetTexelIndex(x, y, z)];
				const uint64 removed = word & want;
				if (removed == 0)
					continue;
				word &= ~want;
				removed_bits[w] = removed;
				const int cnt = sPopCount(removed);
				total_removed += uint(cnt);
				dirty_words |= uint64(1) << w;

				// 8³ chunk 摘要（整字落在一个 8³ chunk 内：字 4 格对齐、chunk 8 格对齐）
				const size_t cidx = size_t(x >> cChunkShift) + size_t(y >> cChunkShift) * cdx + size_t(z >> cChunkShift) * chunk_plane;
				const int cs = int(mChunkSolid[cidx]) - cnt;
				mChunkSolid[cidx] = uint16(cs < 0? 0 : cs);
				for (int yy = 0; yy < 4; ++yy)
				{
					const int lc = sPopCount(removed & sWordAxisMasks.mMask[1][yy]);
					if (lc == 0)
						continue;
					const size_t li = cidx * cChunkSize + ((y + yy) & cChunkMask);
					const int ls = int(mChunkLayerSolid[li]) - lc;
					mChunkLayerSolid[li] = uint8(ls < 0? 0 : ls);
				}
			}

	if (total_removed == 0)
		return 0;

	mNumSolidVoxels -= total_removed;

	// ---- 2. 紧界：每个脏 8³ chunk 增量收缩一次 ----
	//   ★ 全扫版每脏 chunk 扫 512 格；增量版只扫「被挖格触及的紧界面」——典型接触面
	//     挖掘每 chunk 只触及 1~2 个面（~64-128 格），未触及的轴零扫描。
	{
		for (int cz = 0; cz < 2; ++cz)
			for (int cy = 0; cy < 2; ++cy)
				for (int cx = 0; cx < 2; ++cx)
				{
					// 收集这个 8³ chunk 的 2×2×2 个 word（16³ 块按 8 对齐 ⇒ 不跨 chunk）
					uint64 rem8[8];
					bool any = false;
					for (int lz = 0; lz < 2; ++lz)
						for (int ly = 0; ly < 2; ++ly)
							for (int lx = 0; lx < 2; ++lx)
							{
								const int w = (cx * 2 + lx) + (cy * 2 + ly) * 4 + (cz * 2 + lz) * 16;
								const uint64 r = removed_bits[w];
								rem8[lx + ly * 2 + lz * 4] = r;
								any |= r != 0;
							}
					if (!any)
						continue;
					const int gx = inBaseX + cx * 8, gy = inBaseY + cy * 8, gz = inBaseZ + cz * 8;
					const size_t cidx = size_t(gx >> cChunkShift) + size_t(gy >> cChunkShift) * cdx + size_t(gz >> cChunkShift) * chunk_plane;
					sShrinkChunkTight(cidx, rem8);
				}
	}

	// ---- 2b) 实心格包围范围：随挖洞**精确收缩**（2026-11-07）----
	//   ★★★ 这是窄相位遍历框（sSolidAABBWorld / sGetCellRange）和 4³ 记账范围的唯一来源。
	//     旧版「只扩不缩」让挖空的形状继续用整网格大小的实心盒和地形相交；全空时甚至
	//     lo>hi 经 8 角点变换又退化回整网格盒 ⇒ 每个物理步都在空区里跑完整套遍历。
	//   16³ 子形状只有 8 个 chunk，合成成本可忽略。
	sRefreshSolidBounds();

	// ---- 3. 表面位图：按 4³ 字**整字批量**重算（2026-11-07）----
	//   旧版对每颗被挖格 × 7 邻格 × 7 次 IsSolid = 每字最多 3136 次位查询。4³ 块号在
	//   C# 侧是整字 ulong.MaxValue ⇒ 每字 64 格全挖，这条路径就是 Unity 现场
	//   Vxl/Imp-Cut-Native 15.14ms/帧 的来源（占 Imp-Cut 的 98.7%）。
	//   现在每个 4³ 字只读 1 + 6 个邻字，用位平移一次算出整字 64 格的「6 邻全实心」掩码：
	//   surface = solid & ~interior —— 与逐格版逐位相同（越界邻字取 0，等价于 IsSolid 越界返 false）。
	//   重算范围 = 本 16³ chunk 的 64 字 + 6 个面的邻字（被挖格的 ±1 邻居只可能落在这些字里）。
	//   块内 64 格一次都不逐格遍历；只有 delta 非 0 时才按 y 平面 popcount 更新摘要计数。
	if (!mSurfaceBits.empty())
	{
		static constexpr uint64 cX0 = 0x1111111111111111ull, cX3 = 0x8888888888888888ull;
		static constexpr uint64 cY0 = 0x000F000F000F000Full, cY3 = 0xF000F000F000F000ull;
		static constexpr uint64 cZ0 = 0x000000000000FFFFull, cZ3 = 0xFFFF000000000000ull;

		auto read_word = [&](int nx, int ny, int nz) -> uint64
		{
			if (nx < 0 || ny < 0 || nz < 0 || nx >= mSizeX || ny >= mSizeY || nz >= mSizeZ)
				return 0;									// 网格外 = 空气
			return occ[GetTexelIndex(nx, ny, nz)];
		};

		auto recompute = [&](int bx, int by, int bz)
		{
			if (bx < 0 || by < 0 || bz < 0 || bx >= mSizeX || by >= mSizeY || bz >= mSizeZ)
				return;
			const uint64 solid = occ[GetTexelIndex(bx, by, bz)];
			// 位序 bit = x | (y<<2) | (z<<4)：x±1 移 1 位、y±1 移 4 位、z±1 移 16 位；
			// 移到字外的那一行/面用相邻 4³ 字的对应边补上（越界字 = 0）。
			const uint64 xp = ((solid >> 1) & ~cX3) | ((read_word(bx + 4, by, bz) & cX0) << 3);
			const uint64 xm = ((solid << 1) & ~cX0) | ((read_word(bx - 4, by, bz) & cX3) >> 3);
			const uint64 yp = ((solid >> 4) & ~cY3) | ((read_word(bx, by + 4, bz) & cY0) << 12);
			const uint64 ym = ((solid << 4) & ~cY0) | ((read_word(bx, by - 4, bz) & cY3) >> 12);
			const uint64 zp = ((solid >> 16) & ~cZ3) | ((read_word(bx, by, bz + 4) & cZ0) << 48);
			const uint64 zm = ((solid << 16) & ~cZ0) | ((read_word(bx, by, bz - 4) & cZ3) >> 48);
			const uint64 new_surf = solid & ~(xp & xm & yp & ym & zp & zm);

			const size_t wi = GetTexelIndex(bx, by, bz);
			const uint64 old_surf = mSurfaceBits[wi];
			const uint64 delta = new_surf ^ old_surf;
			if (delta == 0)
				return;
			mSurfaceBits[wi] = new_surf;
			const uint64 gained = delta & new_surf;
			const uint64 lost = delta & old_surf;
			// 4 对齐的 4³ 字整个落在同一个 8³ chunk 内 ⇒ cidx 对 64 格相同
			const size_t cidx = size_t(bx >> cChunkShift) + size_t(by >> cChunkShift) * cdx + size_t(bz >> cChunkShift) * chunk_plane;
			const int net = sPopCount(gained) - sPopCount(lost);
			const int stotal = int(mChunkSurface[cidx]) + net;
			mChunkSurface[cidx] = uint16(stotal < 0? 0 : stotal);
			for (int yy = 0; yy < 4; ++yy)
			{
				const uint64 plane = sWordAxisMasks.mMask[1][yy];
				const int lnet = sPopCount(gained & plane) - sPopCount(lost & plane);
				if (lnet == 0)
					continue;
				const size_t li = cidx * cChunkSize + ((by + yy) & cChunkMask);
				const int ltotal = int(mChunkLayerSurface[li]) + lnet;
				mChunkLayerSurface[li] = uint8(ltotal < 0? 0 : ltotal);
			}
		};

		// 本 chunk 的 64 字
		for (int wz = 0; wz < 4; ++wz)
			for (int wy = 0; wy < 4; ++wy)
				for (int wx = 0; wx < 4; ++wx)
					recompute(inBaseX + wx * 4, inBaseY + wy * 4, inBaseZ + wz * 4);
		// 六个面的邻字（各 16 个）；角/边外的字不可能是被挖格的直接邻居，不用重算
		for (int a = 0; a < 4; ++a)
			for (int b = 0; b < 4; ++b)
			{
				recompute(inBaseX - 4, inBaseY + a * 4, inBaseZ + b * 4);
				recompute(inBaseX + 16, inBaseY + a * 4, inBaseZ + b * 4);
				recompute(inBaseX + a * 4, inBaseY - 4, inBaseZ + b * 4);
				recompute(inBaseX + a * 4, inBaseY + 16, inBaseZ + b * 4);
				recompute(inBaseX + a * 4, inBaseY + b * 4, inBaseZ - 4);
				recompute(inBaseX + a * 4, inBaseY + b * 4, inBaseZ + 16);
			}
	}

	// ---- 4. 质量属性：按 4³ 字批量累加（2026-11-07，块内 64 格零逐格遍历）----
	//   需要的只有 7 个标量：cnt / Σr_k / Σr_k²（k = 0,1,2）。
	//   r_k 在 4³ 字内是等差：r_k(d) = c_k + vs*d（c = 块角格心，d = 0..3）。
	//   · 整字（4³ 块记账 ⇒ ulong.MaxValue）用闭式一次算出，块内一次都不逐格：
	//       Σd = 16*(0+1+2+3) = 96，Σd² = 16*(0+1+4+9) = 224
	//       Σr  = 64*c + vs*96
	//       Σr² = 64*c² + 2*c*vs*96 + vs²*224
	//   · 稀疏字（单格记账 / 边缘块）用 ctz 只走实际置位的位，不再固定扫 64 次。
	{
		const float cell_volume = Cubed(mVoxelSize);
		const float cell_mass = cell_volume * mDensity;
		const float cell_inertia = cell_mass * Square(mVoxelSize) / 6.0f;
		const double vs = double(mVoxelSize);
		double cnt = 0.0, sum_r[3] = { 0.0, 0.0, 0.0 }, sum_r2[3] = { 0.0, 0.0, 0.0 };

		for (int w = 0; w < 64; ++w)
		{
			if (((dirty_words >> w) & 1) == 0)
				continue;
			const uint64 bits = removed_bits[w];
			const int wx = w & 3, wy = (w >> 2) & 3, wz = w >> 4;
			const int gx = inBaseX + wx * 4, gy = inBaseY + wy * 4, gz = inBaseZ + wz * 4;
			const Vec3 c = GetVoxelCenter(gx, gy, gz);
			const double c3[3] = { double(c.GetX()), double(c.GetY()), double(c.GetZ()) };

			if (bits == ~uint64(0))
			{
				// 整 4³ 块 64 格全挖：闭式，块内零逐格
				constexpr double n = 64.0, Sd = 96.0, Sd2 = 224.0;
				cnt += n;
				for (int k = 0; k < 3; ++k)
				{
					sum_r[k] += n * c3[k] + vs * Sd;
					sum_r2[k] += n * c3[k] * c3[k] + 2.0 * c3[k] * vs * Sd + vs * vs * Sd2;
				}
				continue;
			}

			// 稀疏：ctz 只遍历实际被挖的格（位 b = dx | dy<<2 | dz<<4）
			uint64 rem = bits;
			while (rem != 0)
			{
				const uint32 lo = uint32(rem);
				const int b = lo != 0? int(CountTrailingZeros(lo)) : 32 + int(CountTrailingZeros(uint32(rem >> 32)));
				rem &= rem - 1;
				const double d3[3] = { double(b & 3), double((b >> 2) & 3), double((b >> 4) & 3) };
				cnt += 1.0;
				for (int k = 0; k < 3; ++k)
				{
					const double r = c3[k] + vs * d3[k];
					sum_r[k] += r;
					sum_r2[k] += r * r;
				}
			}
		}

		for (int k = 0; k < 3; ++k)
			mMassCenterSum[k] -= sum_r[k];
		if (cell_mass > 0.0f)
		{
			const double rsq_sum = sum_r2[0] + sum_r2[1] + sum_r2[2];
			for (int k = 0; k < 3; ++k)
				mInertiaDiag[k] -= cnt * double(cell_inertia) + double(cell_mass) * (rsq_sum - sum_r2[k]);
		}
		mMassProperties.mMass = float(mNumSolidVoxels) * cell_mass;
		if (mNumSolidVoxels > 0 && cell_mass > 0.0f)
		{
			mMassProperties.mInertia = Mat44::sZero();
			mMassProperties.mInertia.SetDiagonal3(Vec3(float(mInertiaDiag[0]), float(mInertiaDiag[1]), float(mInertiaDiag[2])));
			mMassProperties.mInertia.SetColumn4(3, Vec4(0, 0, 0, 1));
		}
		else
		{
			mMassProperties.mInertia = Mat44::sIdentity();
		}
		// 实心包围范围已在 2b) 由 sRefreshSolidBounds() 精确收缩，这里不再「只扩不缩」。
	}

	// 整批只换一次数据版本（逐格路径是每格换一次）
	mVersion = sNextVoxelDataVersion.fetch_add(1, memory_order_relaxed);
#ifdef JPH_DEBUG_RENDERER
	mGeometry = nullptr;
	mGeometryVersion = uint(-1);
#endif // JPH_DEBUG_RENDERER

	return total_removed;
}

//////////////////////////////////////////////////////////////////////////////////////////
// 查询接口
//////////////////////////////////////////////////////////////////////////////////////////

const PhysicsMaterial *VoxelShape::GetMaterial(const SubShapeID &inSubShapeID) const
{
	// ★ 挂了逐体素材质数据时：**按格**返回材质（SubShapeID 里编码了体素坐标）。
	//   摩擦 / 弹性本身不在 PhysicsMaterial 基类里，而是由 VoxelMaterial 带上，再经
	//   PhysicsSystem 的 Combine 回调读出（见 sInstallMaterialCombine）。
	//   没挂数据 ⇒ 退回"整块形状共用一个材质"的旧行为。
	if (HasMaterialData())
	{
		uint x, y, z;
		DecodeSubShapeID(inSubShapeID, x, y, z);
		uint id = GetMaterialId(int(x), int(y), int(z));
		if (id != 0 && id < 256)
			return &sGetVoxelMaterialTable()[id];		// 表指针本身是原子的，见 sVoxelMaterials
	}
	return mMaterial.GetPtr();
}

float VoxelShape::GetContactStrengthAtCell(int inX, int inY, int inZ) const
{
	// ★ 与 GetContactStrength 同一套口径，只是**跳过** SubShapeID 的编码/解码：
	//   逐格枚举里 x/y/z 本来就在手上，绕一趟 EncodeSubShapeID → DecodeSubShapeID 是白费。
	// ★ 没有逐体素材质数据（老区块 / 没挂材质）⇒ 用全局表 0 号的强度：宁可给默认硬度，
	//   也不要返回 0（0 会让「受冲量 > 区域承载」永远成立 ⇒ 一碰就碎）。
	if (HasMaterialData())
	{
		uint id = GetMaterialId(inX, inY, inZ);
		if (id < 256)
			return sGetVoxelMaterialTable()[id].mStrength;
	}
	return sGetVoxelMaterialTable()[0].mStrength;
}

float VoxelShape::GetContactStrength(const SubShapeID &inSubShapeID) const
{
	// ★ 逐格查材质表：编号 → VoxelMaterial::mStrength。走的是和 GetMaterial 同一条解码路径
	//   （SubShapeID 里编了体素坐标），所以「这一击撞到哪些格、每格多硬」在 native 侧就有答案。
	// ★ 没有逐体素材质数据（老区块 / 没挂材质）⇒ 用全局表 0 号的强度：宁可给默认硬度，
	//   也不要返回 0（0 会让「受冲量 > 区域承载」永远成立 ⇒ 一碰就碎）。
	if (!HasMaterialData())
		return sGetVoxelMaterialTable()[0].mStrength;

	uint x, y, z;
	DecodeSubShapeID(inSubShapeID, x, y, z);
	return GetContactStrengthAtCell(int(x), int(y), int(z));
}

void VoxelShape::sSetMaterialTable(const float *inFriction, const float *inRestitution, const float *inStrength, int inCount)
{
	int n = inCount < 256 ? inCount : 256;

	// ★ inStrength 允许为 nullptr（老入口 / 只改摩擦弹性）：此时强度列保持原值。
	//   用 lambda 收敛判空，后面三处（短路比较 / 拷旧值 / 填新值）都不用各写一遍。
	auto strengthAt = [inStrength](int inIndex, float inFallback)
	{
		return inStrength != nullptr ? inStrength[inIndex] : inFallback;
	};

	// ★★ 短路：内容没变就什么都不做（不分配、不换指针）。
	//   体素是随时增删的 —— 上层很自然的写法是"改完方块就整表重导一遍"（VxlMaterialTable.ExportPhysics()），
	//   而那种调用里**绝大多数时候表内容是一样的**。写时复制的前提是"换表是稀有的"，
	//   这条短路把它重新变回零分配，否则每次编辑都 new 一张 6 KB 且永不释放（旧表故意不回收）。
	//   顺带的好处：GetMaterial() 返回的指针保持稳定，上层拿它做缓存 / 按指针比较也不会被打断。
	//   语义上无变化：只覆盖前 n 项 ⇒ 前 n 项相同就等于整张表相同（n 之后本来就不动）。
	{
		const VoxelMaterial *current = sVoxelMaterialTable.load(memory_order_acquire);
		bool unchanged = true;
		for (int i = 0; i < n && unchanged; i++)
			unchanged = current[i].mFriction == inFriction[i] && current[i].mRestitution == inRestitution[i]
				&& current[i].mStrength == strengthAt(i, current[i].mStrength);
		if (unchanged)
			return;
	}

	// 写时复制：新表 -> 拷旧值（超过 inCount 的项保持原样，与旧语义一致）-> 填新值 -> 发布。
	// 读侧没有锁、也不会有半张表：它要么看到旧指针（旧表仍然完整）、要么看到新指针（新表已经填好）。
	VoxelMaterial *table = new VoxelMaterial[256];
	const VoxelMaterial *old = sVoxelMaterialTable.load(memory_order_relaxed);
	for (int i = 0; i < 256; i++)
	{
		table[i].mFriction = old[i].mFriction;
		table[i].mRestitution = old[i].mRestitution;
		table[i].mStrength = old[i].mStrength;
	}
	for (int i = 0; i < n; i++)
	{
		table[i].mFriction = inFriction[i];
		table[i].mRestitution = inRestitution[i];
		table[i].mStrength = strengthAt(i, old[i].mStrength);
	}
	// ★ SetEmbedded 不能少（否则 RefConst 的一次 AddRef + Release 会 delete 掉这张表）：
	//   理由见上面 sVoxelMaterialsEmbedded 的说明。
	for (int i = 0; i < 256; i++)
		table[i].SetEmbedded();

	sVoxelMaterialTable.store(table, memory_order_release);	// 旧表不释放：可能还有线程握着它
}

const VoxelMaterial *VoxelShape::sGetVoxelMaterial(uint inId)
{
	return &sGetVoxelMaterialTable()[inId & 255u];
}

// ---- 按体素材质组合摩擦 / 弹性 ----
// Jolt 默认只认 Body 上的摩擦/弹性（一个刚体一个值）；这两个回调能拿到 SubShapeID，
// 所以体素形状可以按"接触发生在哪一格"取该格方块的摩擦/弹性。
// 组合规则与 Jolt 默认保持同形（摩擦取几何平均、弹性取较大值），只是把取值来源换成材质。

/// 取"这一侧"的摩擦：体素形状且挂了材质数据 ⇒ 用该格方块的摩擦；否则用 Body 的值
static float sVoxelFrictionOf(const Body &inBody, const SubShapeID &inSubShapeID)
{
	const Shape *shape = inBody.GetShape();
	if (shape != nullptr && shape->GetSubType() == EShapeSubType::User1)
	{
		const VoxelShape *voxel = static_cast<const VoxelShape *>(shape);
		if (voxel->HasMaterialData())
		{
			uint x, y, z;
			voxel->DecodeSubShapeID(inSubShapeID, x, y, z);
			uint id = voxel->GetMaterialId(int(x), int(y), int(z));
			if (id != 0)
				return VoxelShape::sGetVoxelMaterial(id)->mFriction;
		}
	}
	return inBody.GetFriction();
}

static float sVoxelRestitutionOf(const Body &inBody, const SubShapeID &inSubShapeID)
{
	const Shape *shape = inBody.GetShape();
	if (shape != nullptr && shape->GetSubType() == EShapeSubType::User1)
	{
		const VoxelShape *voxel = static_cast<const VoxelShape *>(shape);
		if (voxel->HasMaterialData())
		{
			uint x, y, z;
			voxel->DecodeSubShapeID(inSubShapeID, x, y, z);
			uint id = voxel->GetMaterialId(int(x), int(y), int(z));
			if (id != 0)
				return VoxelShape::sGetVoxelMaterial(id)->mRestitution;
		}
	}
	return inBody.GetRestitution();
}

static float sCombineFrictionVoxel(const Body &inBody1, const SubShapeID &inSubShapeID1,
								   const Body &inBody2, const SubShapeID &inSubShapeID2)
{
	return sqrt(sVoxelFrictionOf(inBody1, inSubShapeID1) * sVoxelFrictionOf(inBody2, inSubShapeID2));
}

static float sCombineRestitutionVoxel(const Body &inBody1, const SubShapeID &inSubShapeID1,
									  const Body &inBody2, const SubShapeID &inSubShapeID2)
{
	return max(sVoxelRestitutionOf(inBody1, inSubShapeID1), sVoxelRestitutionOf(inBody2, inSubShapeID2));
}

void VoxelShape::sInstallMaterialCombine(PhysicsSystem &ioSystem)
{
	ioSystem.SetCombineFriction(sCombineFrictionVoxel);
	ioSystem.SetCombineRestitution(sCombineRestitutionVoxel);
}

Vec3 VoxelShape::GetSurfaceNormal(const SubShapeID &inSubShapeID, Vec3Arg inLocalSurfacePosition) const
{
	// SubShapeID 里编码了体素坐标，解出来就知道是哪一个格子
	uint x, y, z;
	DecodeSubShapeID(inSubShapeID, x, y, z);

	if (x < uint(mSizeX) && y < uint(mSizeY) && z < uint(mSizeZ) && IsSolid(int(x), int(y), int(z)))
	{
		Vec3 normal;
		// inLocalSurfacePosition 在"质心局部空间"，网格解析要的是盒心坐标
		if (FindClosestVoxelFace(int(x), int(y), int(z), inLocalSurfacePosition + mCenterOfMass, normal))
			return normal;
	}

	// 解码失败，或者该格被实心邻居完全包围（点在实体深处，没有可以"推出去"的面）：退化成朝上
	return Vec3::sAxisY();
}

void VoxelShape::GetSubmergedVolume(Mat44Arg inCenterOfMassTransform, Vec3Arg inScale, const Plane &inSurface, float &outTotalVolume, float &outSubmergedVolume, Vec3 &outCenterOfBuoyancy JPH_IF_DEBUG_RENDERER(, RVec3Arg inBaseOffset)) const
{
	JPH_PROFILE_FUNCTION();

	// 单格在目标空间里的体积 = vs³ * |sx * sy * sz|，总排水体积 = 实心格数 * 它。
	// ★ outTotalVolume 与 outSubmergedVolume 必须**同一口径**（都含缩放），否则 scale != 1 时
	//   "淹没比例 = 淹没体积 / 总体积"会差一个 |scale| 倍。Jolt 的 Body 固定传 scale = 1
	//   （见 Body::GetSubmergedVolume），所以实体刚体路径上没有差别。
	// ★ 这一句顺带触发惰性统计（EnsureDerived）⇒ 下面的实心包围范围已经可用，可以拿它剪枝。
	Vec3 abs_scale = inScale.Abs();
	float cell_volume = Cubed(mVoxelSize) * abs_scale.GetX() * abs_scale.GetY() * abs_scale.GetZ();
	outTotalVolume = float(GetNumSolidVoxels()) * cell_volume;
	outSubmergedVolume = 0.0f;
	outCenterOfBuoyancy = Vec3::sZero();

	if (mOccupancy == nullptr || mVoxelSize <= 0.0f)
		return;

	// 遍历范围 = 实心格包围范围（GetSolidMin/Max 在没统计过时退化成整张网格，只影响速度不影响结果）
	const int min_x = max(0, GetSolidMin(0)), max_x = min(mSizeX - 1, GetSolidMax(0));
	const int min_y = max(0, GetSolidMin(1)), max_y = min(mSizeY - 1, GetSolidMax(1));
	const int min_z = max(0, GetSolidMin(2)), max_z = min(mSizeZ - 1, GetSolidMax(2));
	if (min_x > max_x || min_y > max_y || min_z > max_z)
		return;

	// 体素形状没有"派生几何"可以整块交给 PolyhedronSubmergedVolumeCalculator，所以逐格算：
	// 每一格都是同一个小立方体，只是位置不同。

	// 逐格求浸没体积与浮心（按体积加权平均）
	float total_submerged = 0.0f;
	Vec3 total_moment = Vec3::sZero();
	// 质心偏移：inCenterOfMassTransform 的原点是质心，网格盒心在 +mCenterOfMass（COM == 0 时原样返回）
	Mat44 voxel_to_target = sToGrid(inCenterOfMassTransform, inScale) * Mat44::sScale(inScale);

	// 格心 = 基点 + 下标 * 步进（与 sVisitSolidVoxelsInBox 同一套仿射分解）：
	// 原来是每格一次 `voxel_to_target * Mat44::sTranslation(格心)` —— 一次完整的 4x4 矩阵乘（16 乘 + 12 加），
	// 而 GetVoxelCenter 是仿射的，整条链可以压成 基点 + 三个步进向量 ⇒ 每格只剩 3 次乘加。
	// ★ 步进向量是**方向**，必须用 Multiply3x3（用点变换会把平移量加进每一步）。
	const Vec3 cell_step[3] =
	{
		voxel_to_target.Multiply3x3(Vec3(mVoxelSize, 0, 0)),
		voxel_to_target.Multiply3x3(Vec3(0, mVoxelSize, 0)),
		voxel_to_target.Multiply3x3(Vec3(0, 0, mVoxelSize)),
	};
	const Vec3 cell_origin = voxel_to_target * (-mHalfExtent) + 0.5f * (cell_step[0] + cell_step[1] + cell_step[2]);

	// 水面在目标空间里的法线 / 常数，以及"一个格子在法线上的投影半径"（= 格心到最远角点在法线上的距离）。
	// 有了它就能在**构造计算器之前**判掉整格在水面上/下的情况：那两种情况下 PolyhedronSubmergedVolumeCalculator
	// 也是早退，但它要先做 8 次点变换 + 8 次 SignedDistance 才发现，这里只要 1 次 dot。
	//
	// 半径 = 0.5 * Σ|n·cell_step[k]|：角点相对格心的偏移就是 ±0.5 * cell_step[k]（见下面的 cell_to_target，
	// 单位立方体的角点在 ±0.5，变换里已经把体素边长算进去了）。
	// ★ 判据必须是**保守**的（半径只能大不能小）：半径小了会把"跨水面"的格子当成"整格在水下"，
	//   那一格的体积会从"按水面切出来的比例"变成 1 —— 浮力静默算错，且没有任何断言能抓到。
	const Vec3 surface_normal = inSurface.GetNormal();
	const float surface_constant = inSurface.GetConstant();
	const float cell_radius = 0.5f * (abs(surface_normal.Dot(cell_step[0]))
									+ abs(surface_normal.Dot(cell_step[1]))
									+ abs(surface_normal.Dot(cell_step[2])));

	// 单位立方体 -> 目标空间里**真实大小的格子**：形状缩放 * 体素边长。
	// ★★ 这里必须把 mVoxelSize 乘进去。只传 inScale 的话，每个格子都会被当成边长 1 米的立方体去切水面：
	//    vs != 1 时水面附近的格子会被一个 1 米的大盒切出"糊在 ±0.5 米范围内"的比例（浮心位置严重偏移），
	//    而报出去的单格体积却是 vs³ * |scale| —— 两者口径不一致。
	Mat44 cell_to_target = sToGrid(inCenterOfMassTransform, inScale) * Mat44::sScale(inScale * mVoxelSize);

	for (int z = min_z; z <= max_z; ++z)
		for (int y = min_y; y <= max_y; ++y)
		{
			// 一行的位偏移（行内不变）：移位后低 4 位就是该行 4 个 x 位置的实心位
			const int shift = ((y & cSubChunkMask) << cSubChunkShift) | ((z & cSubChunkMask) << (2 * cSubChunkShift));

			int x = min_x;
			while (x <= max_x)
			{
				// 这 4 格共用一个字（= 它们所属的那个 4x4x4 子块）；整组为空就一次跳过 4 格。
				// ★ 读的是**实时**的占用位图，就地编辑后不需要 Recalculate 也立刻生效。
				const int x_base = x & ~cSubChunkMask;
				const int x_end = min(max_x, x_base + cSubChunkMask);
				const uint64 word = GetWord(x_base, y, z);
				if (word == 0)
				{
					x = x_end + 1;
					continue;
				}

				for (; x <= x_end; ++x)
				{
					if (((word >> (shift + (x & cSubChunkMask))) & 1) == 0)
						continue;									// 空格：不排水

					const Vec3 center = cell_origin + float(x) * cell_step[0] + float(y) * cell_step[1] + float(z) * cell_step[2];
					const float distance = surface_normal.Dot(center) + surface_constant;

					// 整格都在水面之上（所有角点的有符号距离都 >= 0，与 AreAllAbove 同一判据）：不排水
					if (distance - cell_radius >= 0.0f)
						continue;

					// 整格都在水面之下（所有角点的距离都 < 0，与 AreAllBelow 同一判据）：
					// 体积 = 一格，浮心 = 格心 —— 与 sGetCellSubmergedVolume 的早退分支逐位相同，但省掉整条裁剪计算
					if (distance + cell_radius < 0.0f)
					{
						total_submerged += cell_volume;
						total_moment += cell_volume * center;
						continue;
					}

					// 跨水面：交给单位立方体的四面体裁剪（变换里已经含体素边长 ⇒ 切的是真实格子）。
					// 注意返回值已经是**真实体积**（因为 cell_to_target 把单位立方体放大成了真实格子），
					// 不能再乘一次 cell_volume。
					Mat44 cell_transform = cell_to_target;
					cell_transform.SetTranslation(center);

					float cell_submerged_volume; // 目标空间里的体积（不是 0..1）
					Vec3 cell_center_of_buoyancy;
					if (sGetCellSubmergedVolume(cell_transform, cell_volume, inSurface, cell_submerged_volume, cell_center_of_buoyancy JPH_IF_DEBUG_RENDERER(, inBaseOffset)))
					{
						total_submerged += cell_submerged_volume;
						total_moment += cell_submerged_volume * cell_center_of_buoyancy;
					}
				}
			}
		}

	outSubmergedVolume = total_submerged;
	outCenterOfBuoyancy = total_submerged > 0.0f? total_moment / total_submerged : Vec3::sZero();
}

#ifdef JPH_DEBUG_RENDERER
void VoxelShape::BuildDebugGeometry(Array<DebugRenderer::Triangle> &outTriangles) const
{
	// 只输出"暴露出来的面"：内部面永远看不到，画出来只是浪费顶点。
	//
	// 每个面的 4 个角点用 0/1 表示"取格子的下界/上界"，顺序保证从外面看是逆时针（CCW）——
	// 否则会被渲染器的背面剔除干掉。
	//
	// 顶点坐标已经把 mVoxelSize 含进去了（Draw 只再乘形状缩放），所以这里直接给局部空间坐标。
	struct FaceDesc
	{
		int				mAxis;			///< 外法线所在的轴
		int				mSign;			///< 外法线方向（+1 / -1）
		int				mCorners[4][3];	///< 4 个角点，0 = 下界，1 = 上界
	};
	static const FaceDesc cFaces[6] =
	{
		{ 0,  1, { { 1, 0, 0 }, { 1, 1, 0 }, { 1, 1, 1 }, { 1, 0, 1 } } },	// +X
		{ 0, -1, { { 0, 0, 1 }, { 0, 1, 1 }, { 0, 1, 0 }, { 0, 0, 0 } } },	// -X
		{ 1,  1, { { 0, 1, 0 }, { 0, 1, 1 }, { 1, 1, 1 }, { 1, 1, 0 } } },	// +Y
		{ 1, -1, { { 0, 0, 1 }, { 0, 0, 0 }, { 1, 0, 0 }, { 1, 0, 1 } } },	// -Y
		{ 2,  1, { { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 }, { 0, 0, 1 } } },	// +Z
		{ 2, -1, { { 0, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 }, { 1, 0, 0 } } }	// -Z
	};

	if (mOccupancy == nullptr)
		return;

	// 限制一次提交的四边形数量，防止网格很大时撑爆内存
	uint num_quads = 0;
	// 网格格数 * 6 是暴露面数的上界（实心格数也是惰性的，调试绘制不值得为它触发一次统计）
	uint64 total_cells = uint64(mSizeX) * uint64(mSizeY) * uint64(mSizeZ);
	uint max_quads = uint(min(total_cells * 6, uint64(cMaxDebugQuads)));
	outTriangles.reserve(2 * max_quads);

	float half = 0.5f * mVoxelSize;

	// 遍历范围 = 实心格包围范围。★ GetSolidMin/Max 在没统计过时**返回整张网格**（不触发扫描），
	// 所以这里不会为了一次调试绘制去做一次惰性统计，只是少剪一点。
	const int min_x = max(0, GetSolidMin(0)), max_x = min(mSizeX - 1, GetSolidMax(0));
	const int min_y = max(0, GetSolidMin(1)), max_y = min(mSizeY - 1, GetSolidMax(1));
	const int min_z = max(0, GetSolidMin(2)), max_z = min(mSizeZ - 1, GetSolidMax(2));

	for (int z = min_z; z <= max_z; ++z)
		for (int y = min_y; y <= max_y; ++y)
		{
			// 一行的位偏移（行内不变）：移位后低 4 位就是该行 4 个 x 位置的实心位
			const int shift = ((y & cSubChunkMask) << cSubChunkShift) | ((z & cSubChunkMask) << (2 * cSubChunkShift));

			int x = min_x;
			while (x <= max_x)
			{
				// 这 4 格共用一个字（= 它们所属的那个 4x4x4 子块）：整组为空就一次跳过 4 格。
				// 空网格（地形区块里绝大部分区域）于是只付每 4 格一次 64 位比较。
				const int x_base = x & ~cSubChunkMask;
				const int x_end = min(max_x, x_base + cSubChunkMask);
				const uint64 word = GetWord(x_base, y, z);
				if (word == 0)
				{
					x = x_end + 1;
					continue;
				}

				for (; x <= x_end; ++x)
				{
					if (((word >> (shift + (x & cSubChunkMask))) & 1) == 0)
						continue;

					Vec3 center = GetVoxelCenter(x, y, z);
					Vec3 lo = center - Vec3::sReplicate(half);
					Vec3 hi = center + Vec3::sReplicate(half);

					for (const FaceDesc &face : cFaces)
					{
						// 邻居实心 => 这个面在内部，不画
						int dx = face.mAxis == 0? face.mSign : 0;
						int dy = face.mAxis == 1? face.mSign : 0;
						int dz = face.mAxis == 2? face.mSign : 0;
						if (!IsVoxelFaceExposed(x, y, z, dx, dy, dz))
							continue;

						if (num_quads >= max_quads)
							return;

						// 4 个角点
						Vec3 v[4];
						for (int i = 0; i < 4; ++i)
							v[i] = Vec3(
								face.mCorners[i][0] == 0? lo.GetX() : hi.GetX(),
								face.mCorners[i][1] == 0? lo.GetY() : hi.GetY(),
								face.mCorners[i][2] == 0? lo.GetZ() : hi.GetZ());

						// 拆成两个三角形（保持 CCW）
						outTriangles.push_back(DebugRenderer::Triangle(v[0], v[1], v[2], Color::sWhite));
						outTriangles.push_back(DebugRenderer::Triangle(v[0], v[2], v[3], Color::sWhite));
						++num_quads;
					}
				}
			}
		}
}

void VoxelShape::Draw(DebugRenderer *inRenderer, RMat44Arg inCenterOfMassTransform, Vec3Arg inScale, ColorArg inColor, [[maybe_unused]] bool inUseMaterialColors, bool inDrawWireframe) const
{
	// 只在体素数据变化过（Recalculate 递增了 mVersion）之后重建一次几何。
	// 注意几何里已经把 mVoxelSize 烘进去了，mGeometry 只和 mVersion 有关，与形状缩放无关。
	if (mGeometry == nullptr || mGeometryVersion != mVersion)
	{
		Array<DebugRenderer::Triangle> triangles;
		BuildDebugGeometry(triangles);
		mGeometry = new DebugRenderer::Geometry(inRenderer->CreateTriangleBatch(triangles), GetLocalBounds());
		mGeometryVersion = mVersion;
	}

	inRenderer->DrawGeometry(inCenterOfMassTransform.PreTranslated(-inScale * mCenterOfMass) * Mat44::sScale(inScale), inColor, mGeometry, DebugRenderer::ECullMode::CullBackFace, DebugRenderer::ECastShadow::On, inDrawWireframe? DebugRenderer::EDrawMode::Wireframe : DebugRenderer::EDrawMode::Solid);
}
#endif // JPH_DEBUG_RENDERER

bool VoxelShape::CastRay(const RayCast &inRay, const SubShapeIDCreator &inSubShapeIDCreator, RayCastResult &ioHit) const
{
	JPH_PROFILE_FUNCTION();

	if (mOccupancy == nullptr || mVoxelSize <= 0.0f)
		return false;

	// 三维 DDA（Amanatides & Woo）：沿格子边界一步一步往前走，只访问射线真正穿过的那几格。
	// 射线已经表达在体素形状的**未缩放局部空间**里，格子 c 在该空间里占据
	// [c * s - H, (c + 1) * s - H]。
	// 质心偏移：射线表达在"质心局部空间"，网格盒心在 +mCenterOfMass 处 => 起点先搬回网格空间
	const Vec3 origin = inRay.mOrigin + mCenterOfMass;
	const Vec3 &direction = inRay.mDirection;
	const float s = mVoxelSize;
	const int size[3] = { mSizeX, mSizeY, mSizeZ };
	const float extent[3] = { mHalfExtent.GetX(), mHalfExtent.GetY(), mHalfExtent.GetZ() };

	// 先和网格包围盒做 slab 裁剪，把 t 限制在 [t_enter, t_exit] 内。
	// 这样 DDA 的步数只取决于"射线在网格内穿过的格子数"，与射线本身有多长无关
	// （否则一条 1000 m 的射线会白走几万步）。
	float t_enter = 0.0f;
	float t_exit = 1.0f;
	for (int axis = 0; axis < 3; ++axis)
	{
		float o = origin[axis];
		float d = direction[axis];
		float lo = -extent[axis];
		float hi = extent[axis];

		if (abs(d) < cDirectionEpsilon)
		{
			// 与该轴平行：起点必须落在板内，否则永远不相交
			if (o < lo || o > hi)
				return false;
		}
		else
		{
			float inv_d = 1.0f / d;
			float t1 = (lo - o) * inv_d;
			float t2 = (hi - o) * inv_d;
			if (t1 > t2)
			{
				float tmp = t1;
				t1 = t2;
				t2 = tmp;
			}
			t_enter = max(t_enter, t1);
			t_exit = min(t_exit, t2);
			if (t_enter > t_exit)
				return false;
		}
	}
	if (t_exit < 0.0f)
		return false;

	// 裁剪之后的起点一定落在网格内
	Vec3 start = origin + t_enter * direction;

	// DDA 的初始状态
	int cell[3];
	int step[3];
	float t_max[3];
	float t_delta[3];
	for (int axis = 0; axis < 3; ++axis)
	{
		float o = start[axis];
		float d = direction[axis];

		// 该坐标属于哪一格
		int c = int(std::floor((o + extent[axis]) / s));
		cell[axis] = Clamp(c, 0, size[axis] - 1);

		if (d > cDirectionEpsilon)
		{
			step[axis] = 1;
			float boundary = (cell[axis] + 1) * s - extent[axis];	// 该格沿该轴的上边界
			t_max[axis] = t_enter + (boundary - o) / d;				// t_max 与 t_enter/t_exit 同基准（相对射线 origin），否则命中 fraction 会少加 t_enter
			t_delta[axis] = s / d;
		}
		else if (d < -cDirectionEpsilon)
		{
			step[axis] = -1;
			float boundary = cell[axis] * s - extent[axis];			// 该格沿该轴的下边界
			t_max[axis] = t_enter + (boundary - o) / d;
			t_delta[axis] = -s / d;
		}
		else
		{
			step[axis] = 0;
			t_max[axis] = FLT_MAX;
			t_delta[axis] = FLT_MAX;
		}
	}

	// 步数上限：射线在网格内最多穿过 sizeX + sizeY + sizeZ 格，这里放宽一点做兜底，
	// 保证任何浮点异常都不会变成死循环。
	int max_iterations = 3 * (mSizeX + mSizeY + mSizeZ) + 8;
	float t = t_enter;
	for (int iteration = 0; iteration < max_iterations; ++iteration)
	{
		// 当前格是实心就命中；起点就在实心格里时 fraction 为 0（与内置凸形状的约定一致）
		if (IsSolid(cell[0], cell[1], cell[2]))
		{
			float fraction = max(t, 0.0f);
			if (fraction < ioHit.mFraction)
			{
				ioHit.mFraction = fraction;
				ioHit.mSubShapeID2 = EncodeSubShapeID(inSubShapeIDCreator, uint(cell[0]), uint(cell[1]), uint(cell[2])).GetID();
				return true;
			}
			return false;
		}

		// 走向下一个格子：挑最先被穿过的那个轴
		int axis = 0;
		if (t_max[1] < t_max[axis]) axis = 1;
		if (t_max[2] < t_max[axis]) axis = 2;

		t = t_max[axis];
		if (t > t_exit)
			break;						// 已经越过网格（或射线末端）
		if (step[axis] == 0)
			break;						// 兜底：理论上到不了这里（该轴 t_max == FLT_MAX）

		t_max[axis] += t_delta[axis];
		cell[axis] += step[axis];
	}

	return false;
}

void VoxelShape::CastRay(const RayCast &inRay, const RayCastSettings &inRayCastSettings, const SubShapeIDCreator &inSubShapeIDCreator, CastRayCollector &ioCollector, const ShapeFilter &inShapeFilter) const
{
	JPH_PROFILE_FUNCTION();

	// 形状过滤器
	if (!inShapeFilter.ShouldCollide(this, inSubShapeIDCreator.GetID()))
		return;

	// 先做一次普通射线查询（上限是收集器的 early out fraction）
	RayCastResult hit;
	hit.mFraction = ioCollector.GetEarlyOutFraction();
	if (CastRay(inRay, inSubShapeIDCreator, hit))
	{
		// 正面命中。起点就在实心格内部时 fraction == 0，这种"从内部打出去"的命中
		// 只有在"把凸体当实心"时才上报。
		if (inRayCastSettings.mTreatConvexAsSolid || hit.mFraction > 0.0f)
		{
			hit.mBodyID = TransformedShape::sGetBodyID(ioCollector.GetContext());
			ioCollector.AddHit(hit);
		}

		// 需要背面命中时，从射线末端反向再打一条，把结果折回原来的参数区间
		if (inRayCastSettings.mBackFaceModeConvex == EBackFaceMode::CollideWithBackFaces && !ioCollector.ShouldEarlyOut())
		{
			float start_fraction = min(1.0f, ioCollector.GetEarlyOutFraction());
			float delta_fraction = hit.mFraction - start_fraction;
			if (delta_fraction < 0.0f)
			{
				RayCast inverted_ray { inRay.mOrigin + start_fraction * inRay.mDirection, delta_fraction * inRay.mDirection };

				RayCastResult inverted_hit;
				inverted_hit.mFraction = 1.0f;
				if (CastRay(inverted_ray, inSubShapeIDCreator, inverted_hit)
					&& inverted_hit.mFraction > 0.0f) // fraction == 0 表示反向射线在物体内部就结束了，不算背面命中
				{
					inverted_hit.mFraction = hit.mFraction + (inverted_hit.mFraction - 1.0f) * delta_fraction;
					inverted_hit.mBodyID = TransformedShape::sGetBodyID(ioCollector.GetContext());
					ioCollector.AddHit(inverted_hit);
				}
			}
		}
	}
}

void VoxelShape::CollidePoint(Vec3Arg inPoint, const SubShapeIDCreator &inSubShapeIDCreator, CollidePointCollector &ioCollector, const ShapeFilter &inShapeFilter) const
{
	// 形状过滤器
	if (!inShapeFilter.ShouldCollide(this, inSubShapeIDCreator.GetID()))
		return;

	// 体素形状没有解析形式的"点在内部"判定，用统一的射线数交点办法：
	// 从该点向上打一条射线，命中奇数个面就是在内部。
	// 对"由很多小方块拼出来的非凸形状"来说，这是最省事、而且天然支持空腔/隧道的做法。
	sCollidePointUsingRayCast(*this, inPoint, inSubShapeIDCreator, ioCollector, inShapeFilter);
}

void VoxelShape::CollideSoftBodyVertices(Mat44Arg inCenterOfMassTransform, Vec3Arg inScale, const CollideSoftBodyVertexIterator &inVertices, uint inNumVertices, int inCollidingShapeIndex) const
{
	JPH_PROFILE_FUNCTION();

	if (mOccupancy == nullptr || mVoxelSize <= 0.0f)
		return;

	// 把顶点变到体素形状的未缩放局部空间，找出它落在哪一格；
	// 如果那一格是实心的，就沿"离该点最近的格子面"把顶点推出去。
	Mat44 grid_transform = sToGrid(inCenterOfMassTransform, inScale);
	Mat44 target_to_voxel = grid_transform.Inversed();
	Vec3 inv_scale(
		1.0f / inScale.GetX(),
		1.0f / inScale.GetY(),
		1.0f / inScale.GetZ());
	float half = 0.5f * mVoxelSize;

	for (CollideSoftBodyVertexIterator v = inVertices, end = inVertices + inNumVertices; v != end; ++v)
		if (v.GetInvMass() > 0.0f)
		{
			// 目标空间 -> 未缩放局部空间
			Vec3 local = (target_to_voxel * v.GetPosition()) * inv_scale;

			// 它所在的格子
			int cx = int(std::floor((local.GetX() + mHalfExtent.GetX()) / mVoxelSize));
			int cy = int(std::floor((local.GetY() + mHalfExtent.GetY()) / mVoxelSize));
			int cz = int(std::floor((local.GetZ() + mHalfExtent.GetZ()) / mVoxelSize));
			if (!IsSolid(cx, cy, cz))
				continue;

			// 相对格心的偏移：绝对值最大的那个分量决定"最近的出射面"
			Vec3 center = GetVoxelCenter(cx, cy, cz);
			Vec3 delta = local - center;

			int axis = 0;
			float best = abs(delta.GetX());
			if (abs(delta.GetY()) > best)
			{
				axis = 1;
				best = abs(delta.GetY());
			}
			if (abs(delta.GetZ()) > best)
			{
				axis = 2;
				best = abs(delta.GetZ());
			}

			float sign = delta[axis] < 0.0f? -1.0f : 1.0f;
			float penetration = half - sign * delta[axis];		// 到该面的穿透深度
			if (penetration <= 0.0f)
				continue;										// 点在格子外面

			// 该面的局部法线与局部接触点
			Vec3 local_normal = axis == 0? Vec3(sign, 0, 0) : axis == 1? Vec3(0, sign, 0) : Vec3(0, 0, sign);
			Vec3 local_contact = center + local_normal * half;

			// 换算到目标空间：接触点直接用变换；法线要先按逆缩放再乘 3x3（非等比缩放时法线不是简单变换）
			Vec3 contact = grid_transform * (local_contact * inScale);
			Vec3 normal = grid_transform.Multiply3x3(local_normal * inv_scale).Normalized();

			if (v.UpdatePenetration(penetration))
				v.SetCollision(Plane::sFromPointAndNormal(contact, normal), inCollidingShapeIndex);
		}
}

void VoxelShape::GetTrianglesStart(GetTrianglesContext &ioContext, [[maybe_unused]] const AABox &inBox, [[maybe_unused]] Vec3Arg inPositionCOM, [[maybe_unused]] QuatArg inRotation, [[maybe_unused]] Vec3Arg inScale) const
{
	// 体素形状没有"派生几何"这一层：窄相位是逐格直接做的，不存在可供迭代的三角形列表，
	// 所以这里只是把上下文清零，GetTrianglesNext 永远返回 0。
	// 需要可视化请用 JPH_DEBUG_RENDERER 下的 Draw()（它画的是暴露出来的格子面）。
	memset(&ioContext, 0, sizeof(ioContext));
}

int VoxelShape::GetTrianglesNext(GetTrianglesContext &ioContext, [[maybe_unused]] int inMaxTrianglesRequested, [[maybe_unused]] Float3 *outTriangleVertices, [[maybe_unused]] const PhysicsMaterial **outMaterials) const
{
	// 永远没有三角形（见 GetTrianglesStart 的说明）
	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
// 窄相位：只遍历"两个形状包围盒重叠区"里的实心格，每格单独走凸-凸通道
//////////////////////////////////////////////////////////////////////////////////////////

void VoxelShape::sCollideConvexVsVoxel(const Shape *inShape1, const Shape *inShape2, Vec3Arg inScale1, Vec3Arg inScale2, Mat44Arg inCenterOfMassTransform1, Mat44Arg inCenterOfMassTransform2, const SubShapeIDCreator &inSubShapeIDCreator1, const SubShapeIDCreator &inSubShapeIDCreator2, const CollideShapeSettings &inCollideShapeSettings, CollideShapeCollector &ioCollector, const ShapeFilter &inShapeFilter)
{
	sNarrowPhaseTimer narrow_phase_timer;				// 调试统计：窄相位耗时
	// 这个函数注册在 (任意形状, 体素) 上：shape2 一定是体素形状
	const VoxelShape *voxel = static_cast<const VoxelShape *>(inShape2);

	// ★ 空形状早退（必须在 sStepPairClaim 之前）：没有实心格 ⇒ 零接触、零记账。
	//   挖空的 16³ child 仍被 broadphase 配对，旧路径先付全局锁 + 全表线性扫 +
	//   格号缓冲分配，才一路走到 sCollideVoxelGrids 的空早退。GetNumSolidVoxels()
	//   走 EnsureDerived()，派生有效时只是一次版本比较。
	if (voxel->GetNumSolidVoxels() == 0)
		return;

	// ★★★ 撞击破坏：这一趟逐格碰撞的材质强度累加（体素在 shape2 侧）——
	//   记进按形状对占位的记录，OnContactAdded / 步进后按形状指针查读。
	StepPairRecord *pair_record = sStepPairClaim(inShape1, inShape2);
	VoxelStrengthSumCollector strength_collector(ioCollector, nullptr, voxel, pair_record);
	CollideShapeCollector &out_collector = strength_collector;

	// 质心偏移：inCenterOfMassTransform2 是"质心"的变换，网格要的是盒心的变换（COM == 0 时原样）
	Mat44 voxel_com2 = voxel->sToGrid(inCenterOfMassTransform2, inScale2);

	// 用 shape1 的包围盒圈出要遍历的格子范围（变换到体素的未缩放局部空间）。
	//
	// ★ 必须先按 mMaxSeparationDistance 外扩：这个设置的语义是"相距不超过 d 也算接触"。
	//   如果不外扩，就会出现"两个形状的包围盒并不重叠、但距离在 d 以内"的情况 —— 遍历集合是空的，
	//   整片接触被静默跳过（测试里的表现是：把物体抬高 0.03 m 再配 mMaxSeparationDistance = 0.1 就查不到任何接触）。
	//   这与 CollideConvexVsTriangles 的做法一致（它也是先 ExpandBy(mMaxSeparationDistance) 再换算到对方空间）。
	AABox shape1_world_bounds = inShape1->GetWorldSpaceBounds(inCenterOfMassTransform1, inScale1);
	shape1_world_bounds.ExpandBy(Vec3::sReplicate(inCollideShapeSettings.mMaxSeparationDistance));
	AABox local_box = sWorldBoxToVoxelLocal(shape1_world_bounds, voxel_com2, inScale2);
	voxel->RecordDebugQueryBox(local_box);		// 调试：面板画"查询框"

	const BoxShape *unit_box = sGetUnitBox();
	const SphereShape *unit_sphere = sUseSphereContacts ? sGetUnitSphere() : nullptr;

	// 格的均匀缩放（= 形状缩放 * 体素边长）。胶囊只支持均匀缩放，所以三轴不等长时整条查询退回逐格。
	Vec3 abs_cell_scale = (inScale2 * voxel->mVoxelSize).Abs();
	float cell_scale = abs_cell_scale.ReduceMin();
	bool uniform_cell_scale = cell_scale > 0.0f && abs_cell_scale.ReduceMax() <= cell_scale * (1.0f + 1.0e-4f);

	if (unit_sphere != nullptr && uniform_cell_scale)
	{
		// ---- 游程合并路径：沿 X 连续、体素值相同的实心格合成一根胶囊 ----
		Vec3 sphere_scale = Vec3::sReplicate(cell_scale);
		voxel->sVisitSolidVoxelRunsInBox(local_box, inScale2, voxel_com2,
			[inShape1, inScale1, inCenterOfMassTransform1, &inSubShapeIDCreator1, voxel, &inSubShapeIDCreator2, &inCollideShapeSettings, &out_collector, &inShapeFilter, unit_sphere, sphere_scale](int inX, int inRunLength, int inY, int inZ, const Mat44 &inFirstCellTransform, Vec3Arg inRunCenter, Vec3Arg inRunAxis)
			{
				// 首格的 SubShapeID：游程内体素值相同，取材质/法线用首格都是对的
				SubShapeIDCreator cell_id = voxel->EncodeSubShapeID(inSubShapeIDCreator2, uint(inX), uint(inY), uint(inZ));

				if (inRunLength == 1)
				{
					// 单格游程：还是内切球（胶囊的圆柱半高必须 > 0，单格没有"段"）
					CollisionDispatch::sCollideShapeVsShape(inShape1, unit_sphere, inScale1, sphere_scale, inCenterOfMassTransform1, inFirstCellTransform, inSubShapeIDCreator1, cell_id, inCollideShapeSettings, out_collector, inShapeFilter);
				}
				else
				{
					// 一根胶囊覆盖整段：局部 Y 轴对齐到游程方向，中心落在游程中点
					const CapsuleShape *capsule = sGetUnitCapsule(inRunLength);
					Mat44 capsule_transform = Mat44::sRotationTranslation(Quat::sFromTo(Vec3::sAxisY(), inRunAxis), inRunCenter);
					CollisionDispatch::sCollideShapeVsShape(inShape1, capsule, inScale1, sphere_scale, inCenterOfMassTransform1, capsule_transform, inSubShapeIDCreator1, cell_id, inCollideShapeSettings, out_collector, inShapeFilter);
				}
			});
		return;
	}

	voxel->sVisitSolidVoxelsInBox(local_box, inScale2, voxel_com2,
		[inShape1, inScale1, inCenterOfMassTransform1, &inSubShapeIDCreator1, voxel, &inSubShapeIDCreator2, &inCollideShapeSettings, &out_collector, &inShapeFilter, unit_box, unit_sphere](int inX, int inY, int inZ, const Mat44 &inCellTransform, Vec3 inCellScale)
		{
			// 该格的 SubShapeID：低若干位是 X，然后是 Y，最后是 Z。破坏系统据此知道该删哪一格。
			SubShapeIDCreator cell_id = voxel->EncodeSubShapeID(inSubShapeIDCreator2, uint(inX), uint(inY), uint(inZ));

			if (unit_sphere != nullptr)
			{
				// 内切球路径（非均匀缩放时的兜底）：半径 = 0.5 * min(缩放分量)（与体素 vs 体素格对筛选
				// 阈值里的"半格"同一口径，保证球不超出格子）。球只支持均匀缩放，所以给 sReplicate 的统一缩放。
				// 产出 1 个点接触（面为空时 ManifoldBetweenTwoFaces 退回单点，接触不会丢）。
				Vec3 sphere_scale = Vec3::sReplicate(inCellScale.Abs().ReduceMin());
				CollisionDispatch::sCollideShapeVsShape(inShape1, unit_sphere, inScale1, sphere_scale, inCenterOfMassTransform1, inCellTransform, inSubShapeIDCreator1, cell_id, inCollideShapeSettings, out_collector, inShapeFilter);
			}
			else
			{
				// 每格就是一个独立的"凸盒子 vs 凸盒子"查询，交给常规凸-凸通道处理
				CollisionDispatch::sCollideShapeVsShape(inShape1, unit_box, inScale1, inCellScale, inCenterOfMassTransform1, inCellTransform, inSubShapeIDCreator1, cell_id, inCollideShapeSettings, out_collector, inShapeFilter);
			}
		});
}

void VoxelShape::sCollideVoxelVsVoxel(const Shape *inShape1, const Shape *inShape2, Vec3Arg inScale1, Vec3Arg inScale2, Mat44Arg inCenterOfMassTransform1, Mat44Arg inCenterOfMassTransform2, const SubShapeIDCreator &inSubShapeIDCreator1, const SubShapeIDCreator &inSubShapeIDCreator2, const CollideShapeSettings &inCollideShapeSettings, CollideShapeCollector &ioCollector, const ShapeFilter &inShapeFilter)
{
	sNarrowPhaseTimer narrow_phase_timer;				// 调试统计：窄相位耗时（含情况 A 的 sCollideVoxelGrids）
	sCallProbeLoopNs = 0;								// 诊断：本调用的候选循环累计耗时清零
	sCallWalkNs = 0;									// 诊断：本调用的遍历行走段累计耗时清零
	sCallSubTests = 0;								// 诊断：本调用的子块测试计数清零
	sCallSubEmpty = 0;								// 诊断：本调用的子块空跳计数清零
	// 这个函数注册在 (体素, 任意形状) 上：shape1 一定是体素形状。
	const VoxelShape *voxel = static_cast<const VoxelShape *>(inShape1);
	const bool other_is_voxel = inShape2->GetSubType() == EShapeSubType::User1;
	const VoxelShape *voxel2 = other_is_voxel ? static_cast<const VoxelShape *>(inShape2) : nullptr;

	// ★ 空形状早退（必须在 sStepPairClaim 之前）：任一边没有实心格 ⇒ 零接触、零记账。
	//   与 sCollideVoxelGrids:3952 的早退同义，只是提前到「占记录 + 分配缓冲」之前，
	//   省掉全局锁、全表线性扫和两块格号缓冲分配。原来 claim 出来的那条记录两侧
	//   strength/count 都是 0，读侧一样得到 0 ⇒ 语义完全等价。
	if (voxel->GetNumSolidVoxels() == 0 || (voxel2 != nullptr && voxel2->GetNumSolidVoxels() == 0))
		return;

	// ★★★ 撞击破坏：这一趟逐格碰撞的材质强度累加 —— **双侧记账**。
	//   体素×体素（情况 A）的记录**内联在 sCollideVoxelGrids 的逐格枚举里做**（见那里的说明：
	//   不受 1024 接触点上限限制、格号直接拿 x/y/z 不用编解码）；
	//   情况 B（凸体×体素）仍走包装层 VoxelStrengthSumCollector（它只记承压，格号只在体素侧）。
	StepPairRecord *pair_record = sStepPairClaim(inShape1, inShape2);
	VoxelStrengthSumCollector strength_collector(ioCollector, nullptr, other_is_voxel ? nullptr : voxel,
		other_is_voxel ? nullptr : pair_record);
	CollideShapeCollector &out_collector = other_is_voxel
		? static_cast<CollideShapeCollector &>(ioCollector)
		: static_cast<CollideShapeCollector &>(strength_collector);

	// 质心偏移：形状侧要盒心变换（COM == 0 时原样返回）
	Mat44 voxel_com1 = voxel->sToGrid(inCenterOfMassTransform1, inScale1);

	// ---- 情况 A：对方也是体素网格 -> 走手写快路径 ----
	//
	// 这是可破坏世界里最常见、也最吃性能的组合（碎块互相撞）。
	// 手写路径不查碰撞分发表、不为任何格子创建形状对象、也不做 GJK/EPA，而且能给出整片接触共用的法线 ——
	// 后者是"两个方块叠在一起能互相推开"的前提（详细原因见 sCollideVoxelGrids）。
	if (other_is_voxel)
	{
		// voxel2 已在入口解出（与这里同一判定）⇒ 直接复用，别再声明同名局部（C4456）
		sCollideVoxelGrids(voxel, inScale1, voxel_com1, inSubShapeIDCreator1,
			voxel2, inScale2, voxel2->sToGrid(inCenterOfMassTransform2, inScale2), inSubShapeIDCreator2,
			inCollideShapeSettings, out_collector, pair_record);
		return;
	}

	// ---- 情况 B：对方是凸形状 / Mesh / HeightField / Plane -> 遍历实心格交给通用分发 ----
	//
	// 这里故意**没有**为每个体素 new 一个形状：所有格子共用同一个进程级单位盒子（sGetUnitBox()，或
	// sUseSphereContacts 时的单位球 sGetUnitSphere()），而沿 X 连续、体素值相同的实心格再合并成
	// 进程级胶囊池里的一根胶囊（sGetUnitCapsule）。所以内存开销与"是否体素形状"无关，恒等于体素数组本身。

	// 见 sCollideConvexVsVoxel：同样要按 mMaxSeparationDistance 外扩，否则"靠近但未重叠"的接触会整片丢失。
	AABox shape2_world_bounds = inShape2->GetWorldSpaceBounds(inCenterOfMassTransform2, inScale2);
	shape2_world_bounds.ExpandBy(Vec3::sReplicate(inCollideShapeSettings.mMaxSeparationDistance));
	AABox local_box = sWorldBoxToVoxelLocal(shape2_world_bounds, voxel_com1, inScale1);
	voxel->RecordDebugQueryBox(local_box);		// 调试：面板画"查询框"

	const BoxShape *unit_box = sGetUnitBox();
	const SphereShape *unit_sphere = sUseSphereContacts ? sGetUnitSphere() : nullptr;

	// 格的均匀缩放（= 形状缩放 * 体素边长）。胶囊只支持均匀缩放，所以三轴不等长时整条查询退回逐格。
	Vec3 abs_cell_scale = (inScale1 * voxel->mVoxelSize).Abs();
	float cell_scale = abs_cell_scale.ReduceMin();
	bool uniform_cell_scale = cell_scale > 0.0f && abs_cell_scale.ReduceMax() <= cell_scale * (1.0f + 1.0e-4f);

	if (unit_sphere != nullptr && uniform_cell_scale)
	{
		// ---- 游程合并路径：沿 X 连续、体素值相同的实心格合成一根胶囊 ----
		Vec3 sphere_scale = Vec3::sReplicate(cell_scale);
		voxel->sVisitSolidVoxelRunsInBox(local_box, inScale1, voxel_com1,
			[inShape2, inScale2, inCenterOfMassTransform2, voxel, &inSubShapeIDCreator1, &inSubShapeIDCreator2, &inCollideShapeSettings, &out_collector, &inShapeFilter, unit_sphere, sphere_scale](int inX, int inRunLength, int inY, int inZ, const Mat44 &inFirstCellTransform, Vec3Arg inRunCenter, Vec3Arg inRunAxis)
			{
				// 首格的 SubShapeID：游程内体素值相同，取材质/法线用首格都是对的
				SubShapeIDCreator cell_id = voxel->EncodeSubShapeID(inSubShapeIDCreator1, uint(inX), uint(inY), uint(inZ));

				if (inRunLength == 1)
				{
					// 单格游程：还是内切球（胶囊的圆柱半高必须 > 0，单格没有"段"）
					CollisionDispatch::sCollideShapeVsShape(unit_sphere, inShape2, sphere_scale, inScale2, inFirstCellTransform, inCenterOfMassTransform2, cell_id, inSubShapeIDCreator2, inCollideShapeSettings, out_collector, inShapeFilter);
				}
				else
				{
					// 一根胶囊覆盖整段：局部 Y 轴对齐到游程方向，中心落在游程中点
					const CapsuleShape *capsule = sGetUnitCapsule(inRunLength);
					Mat44 capsule_transform = Mat44::sRotationTranslation(Quat::sFromTo(Vec3::sAxisY(), inRunAxis), inRunCenter);
					CollisionDispatch::sCollideShapeVsShape(capsule, inShape2, sphere_scale, inScale2, capsule_transform, inCenterOfMassTransform2, cell_id, inSubShapeIDCreator2, inCollideShapeSettings, out_collector, inShapeFilter);
				}
			});
		return;
	}

	voxel->sVisitSolidVoxelsInBox(local_box, inScale1, voxel_com1,
		[inShape2, inScale2, inCenterOfMassTransform2, voxel, &inSubShapeIDCreator1, &inSubShapeIDCreator2, &inCollideShapeSettings, &out_collector, &inShapeFilter, unit_box, unit_sphere](int inX, int inY, int inZ, const Mat44 &inCellTransform, Vec3 inCellScale)
		{
			SubShapeIDCreator cell_id = voxel->EncodeSubShapeID(inSubShapeIDCreator1, uint(inX), uint(inY), uint(inZ));
			if (unit_sphere != nullptr)
			{
				// 内切球路径（非均匀缩放时的兜底，同 sCollideConvexVsVoxel 的说明）：每格 1 个点接触
				Vec3 sphere_scale = Vec3::sReplicate(inCellScale.Abs().ReduceMin());
				CollisionDispatch::sCollideShapeVsShape(unit_sphere, inShape2, sphere_scale, inScale2, inCellTransform, inCenterOfMassTransform2, cell_id, inSubShapeIDCreator2, inCollideShapeSettings, out_collector, inShapeFilter);
			}
			else
			{
				CollisionDispatch::sCollideShapeVsShape(unit_box, inShape2, inCellScale, inScale2, inCellTransform, inCenterOfMassTransform2, cell_id, inSubShapeIDCreator2, inCollideShapeSettings, out_collector, inShapeFilter);
			}
		});
}

// ---- 2x2x2 邻域分类的预计算表（sNormalMode == 2 专用）----
//
// 分类 = 把 8 位实心掩码拆成"整面 / 同轴棱 / 单格"这三种基本块。这件事**只依赖掩码本身**：
// 与 b 落在格子里的哪个位置、格长、变换都无关 ⇒ 256 种状态可以在编译期全部展开。
// 运行期于是从"最多 5 轮 x 18 次位扫描（约 90 次位运算）"降到"一次查表 + 逐块向量运算"。
//
//   ★ 用 constexpr 生成而不是手写常量表：拆块规则只存在一份，改贪心顺序时表自动跟着变，
//     不会出现"表与代码失配"这种只能靠对拍才能发现的 bug。
//   ★ mCount == 0 表示"这张表不管"：0 格 / 8 格走全局轴，>= 6 格走"推向空位"，
//     两者仍由原来的分支处理（它们各有一次 while 扫描，那是 O(1) 且不常走）。
//
// 位序：bit = x | (y << 1) | (z << 2)，与内层的 mask4 一致。
enum ENbBlockType : uint8
{
	cNbFace,			// 4 格整面 -> 方向 = 面法线（常量）
	cNbEdge,			// 2 格同轴 -> 方向 = 到那条线段最近点的方向（胶囊语义）
	cNbSphere,			// 1 格     -> 方向 = 离开格心的方向（球语义）
};

struct NbBlock
{
	uint8	mType;		// ENbBlockType
	uint8	mA;			// face: 0..5；edge: 端点 bit；sphere: bit
	uint8	mB;			// 仅 edge 用：另一个端点 bit
	uint8	mAxis;		// 仅 edge 用：两个端点相差的那根轴（0=x 1=y 2=z）
};

struct NbPlan
{
	uint8	mCount;		// 0 = 不适用（见上）
	NbBlock	mBlock[5];	// 1~5 格最多拆成 5 块（每块至少占 1 格）
};

static constexpr int cNbFaceBits[6] = { 0x55, 0xAA, 0x33, 0xCC, 0x0F, 0xF0 };
static constexpr int cNbEdgeMask[12] =
{
	0x03, 0x0C, 0x30, 0xC0,		// 沿 x 的四条棱（y, z 各取 0/1）
	0x05, 0x0A, 0x50, 0xA0,		// 沿 y 的四条棱（x, z 各取 0/1）
	0x11, 0x22, 0x44, 0x88,		// 沿 z 的四条棱（x, y 各取 0/1）
};
static constexpr int cNbEdgeBits[12][2] =
{
	{ 0, 1 }, { 2, 3 }, { 4, 5 }, { 6, 7 },
	{ 0, 2 }, { 1, 3 }, { 4, 6 }, { 5, 7 },
	{ 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 },
};

/// 贪心拆块：整面 > 同轴棱 > 单格。**顺序必须与运行期的老实现逐条一致**，否则浮点累加顺序变了，
/// acc4 会出现 1 ulp 级差异（物理上会被放大，所以这里按"逐位相同"对待）。
constexpr NbPlan sBuildNbPlan(int inMask)
{
	NbPlan plan {};
	int n = 0;
	for (int i = 0; i < 8; ++i)
		n += (inMask >> i) & 1;

	// 0 格（邻域全空）/ 8 格（完全埋住）/ 6~7 格（推向空位）都不归这张表管
	if (n == 0 || n >= 6)
		return plan;

	int rest = inMask;
	while (rest != 0 && plan.mCount < 5)
	{
		int face = -1;
		for (int f = 0; f < 6; ++f)
			if ((rest & cNbFaceBits[f]) == cNbFaceBits[f])
			{
				face = f;
				break;
			}
		if (face >= 0)
		{
			plan.mBlock[plan.mCount++] = NbBlock { cNbFace, uint8(face), 0 };
			rest &= ~cNbFaceBits[face];
			continue;
		}

		int edge = -1;
		for (int e = 0; e < 12; ++e)
			if ((rest & cNbEdgeMask[e]) == cNbEdgeMask[e])
			{
				edge = e;
				break;
			}
		if (edge >= 0)
		{
			// cNbEdgeBits 按 x/y/z 各四条排列 ⇒ 轴号 = edge / 4（同轴两格只差这一位）
			plan.mBlock[plan.mCount++] = NbBlock { cNbEdge, uint8(cNbEdgeBits[edge][0]), uint8(cNbEdgeBits[edge][1]), uint8(edge >> 2) };
			rest &= ~cNbEdgeMask[edge];
			continue;
		}

		int bit = 0;
		while ((rest & (1 << bit)) == 0)
			++bit;
		plan.mBlock[plan.mCount++] = NbBlock { cNbSphere, uint8(bit), 0 };
		rest &= ~(1 << bit);
	}
	return plan;
}

struct NbPlanTable
{
	NbPlan	mPlan[256];
};

constexpr NbPlanTable sBuildNbPlanTable()
{
	NbPlanTable table {};
	for (int m = 0; m < 256; ++m)
		table.mPlan[m] = sBuildNbPlan(m);
	return table;
}

/// 256 项 x 21 字节 = 5.4 KB，常驻只读段；实际热点只会命中少数几项。
static constexpr NbPlanTable sNbPlans = sBuildNbPlanTable();

// ---- 6~7 格（空位 1~2 个）的预计算表 ----
//
// 空位本来就是掩码补集，e0 / e1 / 是否同轴全是掩码的纯函数 ⇒ 一起在编译期展开，
// 运行期不再 while 扫两次（最多 16 次位测试）。
//
//   ★ mCount == 0：没有空位（n4 == 8，整格埋住 -> 全局轴）。
//   ★ n4 == 0（补集全满）在这个位置**不可达**：2x2x2 一定包含候选格 a 自己
//     （lo4[k] ∈ {ac[k]-1, ac[k]}），而 a 已经过了 IsSolid 筛选 ⇒ 至少 1 格实心。
//   ★ mCoaxial == 1：两个空位只差一位 = 同一根轴上的线段（空槽 / 凹槽），可走 2D。
struct NbEmptyPlan
{
	uint8	mCount;		// 0 = 无空位；1 = 单空位；2 = 两个空位
	uint8	mCoaxial;	// 1 = 两个空位同轴（用 mAxis）
	uint8	mAxis;		// 同轴时：0=x 1=y 2=z
	uint8	mBit[2];	// 空位 bit，升序（与旧版 e0 -> e1 的扫描顺序一致）
};

constexpr NbEmptyPlan sBuildEmptyPlan(int inMask)
{
	NbEmptyPlan plan {};
	int empty = (~inMask) & 0xFF;
	for (int b = 0; b < 8 && plan.mCount < 2; ++b)
		if (empty & (1 << b))
			plan.mBit[plan.mCount++] = uint8(b);

	if (plan.mCount == 2)
	{
		int diff = plan.mBit[0] ^ plan.mBit[1];
		if ((diff & (diff - 1)) == 0)			// 只有一位不同 = 同轴
		{
			plan.mCoaxial = 1;
			plan.mAxis = uint8(diff == 1? 0 : (diff == 2? 1 : 2));
		}
	}
	return plan;
}

struct NbEmptyPlanTable
{
	NbEmptyPlan	mPlan[256];
};

constexpr NbEmptyPlanTable sBuildEmptyPlanTable()
{
	NbEmptyPlanTable table {};
	for (int m = 0; m < 256; ++m)
		table.mPlan[m] = sBuildEmptyPlan(m);
	return table;
}

/// 256 项 x 5 字节 = 1.25 KB
static constexpr NbEmptyPlanTable sEmptyPlans = sBuildEmptyPlanTable();

/// 全局轴的选轴 / 定向统一在 sCollideVoxelGrids 的 1c) 节：
///   1. axis = 实心盒心(shape2) − 接触区中心（= 双方实心格 AABB 交集的中心）
///   2. 上式为零向量 ⇒ axis = 实心盒心(shape2) − 实心盒心(shape1)
///   3. 仍为零 ⇒ +Y
/// 【已删 sFixAxisSign】（探链定符号）与【已删 SAT 轴采用】：旧的两级修正都会覆盖上面的
/// 公式结果，按"轴只来自这两条规则"的约定一并移除（sAxisFixStats 保留，恒为 0）。


void VoxelShape::sCollideVoxelGrids(const VoxelShape *inVoxel1, Vec3Arg inScale1, Mat44Arg inCenterOfMassTransform1, const SubShapeIDCreator &inSubShapeIDCreator1,
									const VoxelShape *inVoxel2, Vec3Arg inScale2, Mat44Arg inCenterOfMassTransform2, const SubShapeIDCreator &inSubShapeIDCreator2,
									const CollideShapeSettings &inCollideShapeSettings, CollideShapeCollector &ioCollector,
									StepPairRecord *inRecord)
{
	// 任何一边没有体素数据，就不可能接触
	if (inVoxel1->mOccupancy == nullptr || inVoxel2->mOccupancy == nullptr)
		return;

	// ★★ 本函数的 inCenterOfMassTransform1/2 名不副实：调用方（sCollideVoxelVsVoxel，见上方
	//     voxel_com1 / voxel2->sToGrid(...)）**已经**用 sToGrid 把 Jolt 传进来的质心变换转成了
	//     「盒心变换」（网格中心系 -> 世界）。GetVoxelCenter 返回的是盒心系坐标，所以本函数
	//     里所有网格空间的数学（OBB / 实心 AABB / 遍历范围 / 跨格映射 / 格心世界坐标）都
	//     **直接用这个入参**，语义就是盒心变换 —— 千万不要再 sToGrid 一次：那是双重转换，
	//     整个网格的感知位置会漂 R*scale*m（mCenterOfMass ≠ 0 的单 shape 碎块回退路径会
	//     SetCenterOfMass(comLocal) ≠ 0），症状 = 接触点杠杆臂全错 + 侧向弹飞 + 穿落
	//     （实测 TinyPieceDrop / TinyPiecesVsEachOther / TinyPieceSpinningDrop 全挂）。
	//     只有把 Jolt 原始的质心变换送进来的入口（外部直接调本函数的新调用点，若有）
	//     才需要先 sToGrid —— 目前没有这样的调用点。

	// ---- 1) 全局轴：**一条与构型无关的公式** ----
	//
	//       axis = 盒心(shape2) − 接触区中心
	//       接触区 = 双方**实心格 AABB** 的交集（世界空间）
	//       交集为空 / 盒心正好落在接触区中心  ⇒ 退回 盒心(shape2) − 盒心(shape1)
	//       连盒心都重合                        ⇒ +Y（"随便挑一根轴"）
	//
	// 为什么是这一条：体素能拼出任意形状，轴必须由"B 相对接触区偏在哪一侧"决定，
	// 而不是由 15 根候选轴里谁的行程更短决定。逐种情形核对，它给的都是同一个语义：
	//   · 方块压在斜坡上 / 陷进地表一格：交集是方块底部那一小片 ⇒ 中心在方块下方 ⇒ 轴朝上；
	//   · 方块整个落在地形实心盒内：交集 = 方块自己 ⇒ 中心 ≈ 方块盒心 ⇒ 退化 ⇒ 盒心差
	//     （只要两盒心不重合就有方向 —— 这正是"被包含"该用的判据）；
	//   · 两块大面积互穿（深穿透）：中心在重叠区中部 ⇒ 轴指向 B 偏出重叠区的那一侧；
	//   · 缝隙（交集为空）：没有接触区，盒心差是唯一可用的方向。
	//   ★ 不需要嵌套门控、不需要厚度投影、不按构型分支 —— 加一条分支就多一种失效方式。
	//
	// ★ 轴可以是任意方向：公式给出的就是带符号的单位向量，下游（投影半径 / 法线）对任意方向都成立。
	//   sGetProjectionRadius 对任意轴给出的都是立方体的精确支撑半径。
	//
	// 派生统计提前到最前：实心包围范围（solid AABB / 洞口条带检测）和表面缓存都要用。
	if (!inVoxel1->sIsDerivedValid())
		inVoxel1->GetNumSolidVoxels();		// 公开包装 = EnsureDerived()
	if (!inVoxel2->sIsDerivedValid())
		inVoxel2->GetNumSolidVoxels();
	if (sInteriorSkipPairs == 2)
	{
		if (inVoxel1->mSurfaceBits.empty())
			inVoxel1->EnsureDerived();
		if (inVoxel2->mSurfaceBits.empty())
			inVoxel2->EnsureDerived();
	}

	// ★★★ 挖空后立刻早退：形状里已经没有实心格，包围盒再大也不可能有接触。
	//   必须放在 SAT / 全局轴 / 遍历范围 / 4³ 记账之前 —— 否则每个空子形状每个物理步
	//   都要把整片空区走一遍（现场 Jolt/Step 恒定 10~15ms/步 就是这些空跑）。
	if (inVoxel1->mNumSolidVoxels == 0 || inVoxel2->mNumSolidVoxels == 0)
		return;

	// ---- 1a) 双方实心格 AABB 的交集（世界空间）= 接触区 ----
	AABox solid1 = sSolidAABBWorld(inVoxel1, inCenterOfMassTransform1, inScale1);
	AABox solid2 = sSolidAABBWorld(inVoxel2, inCenterOfMassTransform2, inScale2);
	AABox meet_world = solid1.Intersect(solid2);

	// ★ 挖空 / 实心盒没有交集 ⇒ 零接触。原逻辑会一路走到 2) 的 sGetCellRange 才返回，
	//   中间 SAT、边界掩码、全局轴、4³ 记账全是空跑（这些正是 Jolt/Step 里那 10~15ms/步）。
	if (!meet_world.IsValid())
		return;

	// ---- 1b) 整体分离量（只做早退与诊断，SAT 的**轴**不再采用）----
	//   ★ 保留 SAT 只为拿到"整网格最短重叠量"这一个标量：分离即无接触（Unity 侧没有推测性接触）。
	//     连续掩码 / 边界细化仍在 SAT 内部生效，只影响这个标量，不影响下面的轴。
	uint dir_mask1 = sRefineBorderMask(inVoxel1, inCenterOfMassTransform1, inScale1, meet_world);
	uint dir_mask2 = sRefineBorderMask(inVoxel2, inCenterOfMassTransform2, inScale2, meet_world);
	VoxelOBB obb1 = sGetGridOBB(inVoxel1->mSizeX, inVoxel1->mSizeY, inVoxel1->mSizeZ, inVoxel1->mVoxelSize, inScale1, inCenterOfMassTransform1, dir_mask1);
	VoxelOBB obb2 = sGetGridOBB(inVoxel2->mSizeX, inVoxel2->mSizeY, inVoxel2->mSizeZ, inVoxel2->mVoxelSize, inScale2, inCenterOfMassTransform2, dir_mask2);
	float second_overlap;
	float grid_overlap = sObbMinSeparationOverlap(obb1, obb2, second_overlap);

	// 诊断用（见 VoxelShape.h）：多线程写 ⇒ 只能是 relaxed 的原子 store
	sLastGridOverlap[0].store(grid_overlap, memory_order_relaxed);
	sLastGridOverlap[1].store(second_overlap, memory_order_relaxed);

	// ★★ 【已删：实心 AABB 交集为空时补厚】原来的推测性接触链（max_separation 外扩 / mExpand /
	//    filter_dist / depth 门控）全部移除，语义回到"分离即无接触"。
	//    参数保留只为签名稳定（Unity 侧 MaxSeparationDistance 本来就没设，一直是 0）。
	(void)inCollideShapeSettings;
	if (grid_overlap < 0.0f)
		return;										// 两个网格整体已经分离，不可能有接触

	// ---- 1c) 全局轴：唯一的一条公式（见 1) 节说明）----
	//
	//   方向 = 盒心(shape2) − 接触区中心；交集为空或该向量退化 ⇒ 盒心(shape2) − 盒心(shape1)。
	//   ★ 两处都用**盒心变换**的平移列（= 网格中心，不是质心）：函数头钉死的契约。
	Vec3 axis_world;
	{
		// ★ 用**实心盒心**（solid AABB 的中心），不是整网格盒心：地块网格的盒心在几何中心，
		//   100 层地形时它在地表下 25 m ⇒ 方块站在地表时"盒心差"朝下，方向全错。
		//   实心 AABB 的中心才代表"这块实心材料在哪"。
		const Vec3 c1 = (solid1.mMin + solid1.mMax) * 0.5f;
		const Vec3 c2 = (solid2.mMin + solid2.mMax) * 0.5f;

		axis_world = c2 - c1;
		if (meet_world.IsValid())
		{
			Vec3 v = c2 - (meet_world.mMin + meet_world.mMax) * 0.5f;
			// 长度阈值：方块盒心正好落在接触区中心时方向无定义 ⇒ 用盒心差。
			// （1e-5 m 的平方 = 1e-10；比这更小的偏移是浮点噪声，归一化后方向随机。）
			if (v.LengthSq() > 1.0e-10f)
				axis_world = v;
		}

		float len = axis_world.Length();
		axis_world = len > 1.0e-6f? axis_world / len : Vec3::sAxisY();

	}

	// ---- 2) 两侧各圈一次"需要遍历的格子范围"，并选**遍历侧** ----
	//
	// 遍历范围源盒 = **双方实心格 AABB 的交集**（世界空间），**不做任何外扩**。
	//
	// ★ 为什么不能用"对方整网格 OBB 的世界盒"：它含全部空格 —— 大物体悬在地表上方时，它的
	//   OBB 会把正下方一整段实心地形和物体自身上方的空气整片圈进来。sGetCellRange 的实心夹紧
	//   只对**本方**生效，一侧范围空掉之后遍历会落到另一侧、照样把整块实心网格走一遍
	//   （离线实测 32^3 物体悬空 5 cm、0 接触仍要 235 us/query，64^3 悬空 0.1 m 要 3.2 ms）。
	//   派生统计无效（从未 EnsureDerived）时 GetSolidMin/Max 回退整网格 ⇒ 行为与旧版一致。
	//
	// ★★ 交集为空 ⇒ sGetCellRange 两侧都返回 false ⇒ 本函数直接 return（零接触）。
	//   这是"删掉空交集补厚"之后**期望**的行为：双方实心盒没有真实交集时，不再靠外扩把
	//   近邻格对圈进来造支撑接触。
	//
	// 注意不要再额外加余量去抠格子边界：sGetCellRange 的边界规则是"正好落在格子边界上的坐标
	// 归给上面那一格"（避免相邻格子重复计数），额外的余量会让边界格子同时被两侧算到，接触数
	// 翻倍、聚合面积也随之翻倍 —— 这会直接破坏"逐格接触的聚合 = 整面接触"这个契约
	// （UnitTests 的 VoxelShapeVsVoxelShape 正是靠它校验的）。两侧的 iter_box 来自**同一个**
	// 世界盒（只是各自变换到局部），所以不存在"两侧各自加余量导致不对称"的问题。
	//
	// ★ 遍历侧取"重叠区格数少"的那一侧：遍历成本 ∝ 遍历的格子数，而"大网格 vs 小方块"这种
	//   最常见的组合里，遍历小方块那一侧能把内层循环的候选查询量砍掉一个数量级以上。
	//   两侧格数一样（对称接触）时保持 shape1 侧，行为与旧版一致。
	//
	// ★ 表面枚举（sInteriorSkipPairs == 2）在选边之前先把表面缓存建好：CountSurfaceCellsInRange
	//   在缓存未建时返回 -1（回退盒格数），而枚举 gate 又会惰性建缓存 —— 同一对形状的第一次
	//   查询与后续查询会选到**不同的遍历侧**，接触数撞截断时集合就不一样了。建一次缓存让选边
	//   从第一次就稳定。
	// （表面缓存 / 派生统计的惰性建块已统一提前到 1) 节开头 —— 见那里的说明。）

	// ---- 1d) 轴到此定稿：方向与符号**都**来自 1c) 的公式，没有任何后续修正 ----
	//   ★【已删 sFixAxisSign】：旧的探链符号修正会**覆盖**公式给出的方向 —— 上次诊断里
	//   "公式给出朝上、探链认可成朝下"就是它。按约定轴只来自两条规则，不再有第二意见。
	// ★★ 【已删：空交集补厚】这里原来做
	//       meet_world.ExpandBy(max_separation + 较大一侧的格长)
	//   把双方实心格 AABB 的交集外扩一圈，目的是让"实心盒还没有真正交上、但格心球已经够近"的
	//   格对也能被圈进遍历范围。问题是：当双方实心盒**完全不相交**（方块静止时允许留 0~2 cm
	//   slop 缝隙）时，这一扩会把空交集补成一个"看起来有效"的盒子，于是查询照样产出接触 ——
	//   也就是在没有任何真实交集的情况下用全局轴硬造支撑。按要求整段删除：
	//   遍历范围直接用交集本身，空交集 ⇒ sGetCellRange 两侧都返回 false ⇒ 本次查询零接触。
	AABox iter_box[2] =
	{
		sWorldBoxToVoxelLocal(meet_world, inCenterOfMassTransform1, inScale1),	// 遍历 shape1 时的范围
		sWorldBoxToVoxelLocal(meet_world, inCenterOfMassTransform2, inScale2)	// 遍历 shape2 时的范围
	};

	int range[2][6];
	bool has_range[2] =
	{
		inVoxel1->sGetCellRange(iter_box[0], range[0][0], range[0][1], range[0][2], range[0][3], range[0][4], range[0][5]),
		inVoxel2->sGetCellRange(iter_box[1], range[1][0], range[1][1], range[1][2], range[1][3], range[1][4], range[1][5])
	};
	if (!has_range[0] && !has_range[1])
		return;

	auto cell_count = [](const int *inRange)
	{
		return double(inRange[3] - inRange[0] + 1) * double(inRange[4] - inRange[1] + 1) * double(inRange[5] - inRange[2] + 1);
	};
	double count[2] = { has_range[0]? cell_count(range[0]) : 0.0, has_range[1]? cell_count(range[1]) : 0.0 };

	// 遍历侧选择改用 **range 内的实心格数**（8^3 chunk 摘要，y 按 1 格分层）：
	// "80^3 球贴着 128^3 地形"时，重叠区里球侧只有几百个实心格、地形侧 80×80×几层全是实心 ——
	// 按盒格数两者相当、会选中地形侧白走一万多格（实测悬空 1 格、0 接触仍要 1.9 ms/query；
	// 球侧只有 ~5 个实心格时也是这个数）。摘要无效时回退盒格数（与旧版一致）。
	// ★ 重叠区很小时（两个 8^3 碎块 = 128 格这种）盒格数不会骗人，摘要查询是纯开销
	//   —— 阈值取 4 个 chunk 的格数。大物体 vs 地形（摘要要解决的场景）远超这个数。
	int solid_count[2] = { int(count[0]), int(count[1]) };
	if (count[0] + count[1] >= 4.0 * double(cChunkVolume))
	{
		// sInteriorSkipPairs == 2（表面枚举）：按**表面格数**选遍历侧 —— 内部格在该过滤下产不出
		// 任何接触，枚举它们的成本纯浪费。球贴地形时地形侧表面只有几千（footprint 那一层）、
		// 球侧表面上万，按实心格数会选错边（球侧 10 万+）。
		if (has_range[0]) solid_count[0] = sInteriorSkipPairs == 2? inVoxel1->CountSurfaceCellsInRange(range[0]) : inVoxel1->CountSolidCellsInRange(range[0]);
		if (has_range[1]) solid_count[1] = sInteriorSkipPairs == 2? inVoxel2->CountSurfaceCellsInRange(range[1]) : inVoxel2->CountSolidCellsInRange(range[1]);
		if (solid_count[0] < 0) solid_count[0] = int(count[0]);
		if (solid_count[1] < 0) solid_count[1] = int(count[1]);
	}


	// ★ 位图大小惩罚：表面过滤（IsVoxelSurface 查 6 邻）与 mode-4 邻域都读**遍历侧**位图，
	//   每格成本随位图大小陡增（实测 2MB 地形 ~0.5us/格 vs 125KB 球 ~0.16us/格，深陷时差 8 倍）。
	//   惩罚 = 位图字节数 / 256KB（~L2 的一半，经验值），下限 1 —— 小形状之间不互相惩罚，
	//   大地形 vs 小形状时把选侧拉向小形状一侧（深陷球 5m：地形侧表面 1 万 x8 仍输给球侧 9 千 x1）。
	auto bitmap_penalty = [](const VoxelShape *inVoxel)
	{
		double bytes = double(inVoxel->mSizeX) * double(inVoxel->mSizeY) * double(inVoxel->mSizeZ) / 8.0;
		double f = bytes / 262144.0;
		return f < 1.0? 1.0 : f;
	};
	double cost[2] = { double(solid_count[0]) * bitmap_penalty(inVoxel1), double(solid_count[1]) * bitmap_penalty(inVoxel2) };

	int iterate_side = (has_range[0] && (!has_range[1] || cost[0] <= cost[1]))? 0 : 1;
	if (sIterateSideOverride != 0)						// 单元测试用：强制某个方向（"两个方向结果一致"的对照）
	{
		int forced = sIterateSideOverride - 1;
		if (has_range[forced])
			iterate_side = forced;
	}
	sStatLastIterSide.store(iterate_side, std::memory_order_relaxed);
	for (int k = 0; k < 6; ++k) sStatLastRange[k] = -1;
	for (int k = 0; k < 6; ++k) sTlsLastRange[k] = -1;
	if (has_range[iterate_side])
	{
		for (int k = 0; k < 6; ++k) sStatLastRange[k] = range[iterate_side][k];
		for (int k = 0; k < 6; ++k) sTlsLastRange[k] = range[iterate_side][k];
	}

	// ---- 3) 两侧的公共常量 ----
	//
	// ★ 问题：meet 盒是**世界空间**的轴对齐盒，某一侧带旋转时它映回该侧局部空间会被撑成
	//   "肥 AABB"—— 世界里几格厚的接触薄板，局部空间变成几十层厚的大盒（斜板的包围盒）。
	//   球落地后翻滚就是这个形态：按肥盒整遍要访问几万格（实测 80^3 球贴地 5.2 ms/query，
	//   轴对齐同姿态 0.3 us），而 4c 的 chunk 剪枝在"松到没边"的候选区间上剪不动。
	//
	// ★ 做法：遍历侧改选**带旋转的一侧**，但不再按肥盒逐格枚举 —— 按"列"枚举（见 4c-pre）：
	//   对垂直于"最贴合世界 Y 的局部轴"的每一列，三条世界轴约束各解一个一维仿射区间求交，
	//   只访问格心落在 meet 盒内的格子（薄板里通常只有 1~3 格/列）。
	//
	// ★ 正确性：接触对的遍历格心必在 meet 盒内（2) 的外扩论证），盒外格子的跨格候选全部被
	//   球面筛选刷掉、本来就不产出接触 ⇒ 枚举集合是原路径的子集、接触集合一致
	//   （UnitTests 的 VoxelShapeRotatedSlabExact 用顺序无关哈希对照验证）。访问**顺序**与盒
	//   遍历不同（环序，由外往内），接触数撞 sMaxContactPoints 截断时保留的仍是整片外围。
	//
	// ★ 门槛：只在一侧轴对齐（地形）、另一侧带旋转、且旋转侧肥盒格数 > 4 chunk 时启用；
	//   两侧都旋转（BitExact 的随机姿态）或都是小物体走原路径。sIterateSideOverride 强制
	//   遍历侧时也不启用（保住"两个方向一致"的对照语义）。sRotatedFastPath = 0 整体关闭（A/B 用）。
	bool aligned_side[2];
	for (int i = 0; i < 2; ++i)
	{
		Mat44Arg t = i == 0? inCenterOfMassTransform1 : inCenterOfMassTransform2;
		aligned_side[i] = true;
		for (int k = 0; k < 3 && aligned_side[i]; ++k)
		{
			Vec3 c = t.GetColumn3(uint(k));
			Vec3 d = c - Vec3(k == 0? 1.0f : 0.0f, k == 1? 1.0f : 0.0f, k == 2? 1.0f : 0.0f);
			if (d.LengthSq() > 1.0e-6f)					// 偏差 > 1e-3 视为带旋转
				aligned_side[i] = false;
		}
	}
	const int rot_side = aligned_side[0] != aligned_side[1]? (aligned_side[0]? 1 : 0) : -1;
	bool use_columns = false;
	// [suspended] skip=2 also enabling column enumeration: after the OBB clip, per-column windows are only a few cells, but the window cell selection and
	// the fat-box enumeration have a contact set difference at the tight meet boundary (RotatedSlabExact 11/16 pose hash mismatch, count equal),
	// root cause not yet found, temporarily restored to disable at skip=2.
	// 不会再出现"整列扫到 meet 盒边界重扫整球"的旧问题（那是因为旧 meet 盒被肥 AABB 撑肥）。
	if (sInteriorSkipPairs != 2
		&& rot_side >= 0 && sRotatedFastPath != 0 && sIterateSideOverride == 0
		&& has_range[rot_side] && count[rot_side] > 4.0 * double(cChunkVolume))
	{
		use_columns = true;
		iterate_side = rot_side;
	}

	// 调试：记录本对查询用的局部盒（编辑器面板画"查询框"）。只记**遍历侧**——它的盒才真正用于
	// 筛选体素；候选侧的盒不参与筛选（候选区间是逐格仿射映射算的），记了只会画出误导性的大框。
	(iterate_side == 0? inVoxel1 : inVoxel2)->RecordDebugQueryBox(iter_box[iterate_side]);

	// ---- 2c) ★★★ B 方案「块抽样」门槛（2026-11-06）----
	//
	// 大重叠（深嵌球、碎块撞击）时逐格窄相位一帧要走几万格（实测 5m 球埋基岩：遍历 6 万格/步，
	// ~100ns/格），是帧率崩掉的根源。这里按**遍历侧重叠区间的格数**把查询分成两档：
	//
	//   · 区间格数 < sBlockSampleMinVolume：小重叠（静止方块、角色脚下）⇒ 走原逐格路径，
	//     下面所有行为与旧版**逐位一致**。
	//   · 区间格数 ≥ 门槛：大重叠 ⇒ 接触和记账都按 2x2x2 块抽样：
	//     遍历侧不再逐格枚举，改为枚举**块角格**（全局偶数下标，块 = [角, 角+1]³ 的 8 格）；
	//     一个块最多产出 **1 个接触点**（代表对 = 块心当"直径 2 格的体素" × 窗口里的候选实心格），
	//     记账则把块内 8 个遍历格 + 窗口里全部候选实心格记全（dedup 去重，承压一颗一次）。
	//     代价（刻意接受）：接触从每格一对变成每块一对（少 8 倍、位置在块心，无偏）；
	//     记账边缘多记一层窗口壳（与 try_record_block 的粗判同一量级的"宁可多记"）。
	//
	// 块的锚点用**全局偶数下标**（不是 range 起点）：chunk 是 8 对齐的，chunk 子盒里的偶数角
	// 恰好把整个 chunk 铺满且跨 chunk 不重不漏 —— chunk 级剪枝（4c）跳掉的 chunk 不会留下
	// "半块没处理"的洞（块角落在被剪 chunk 里、格落在候选 chunk 里的块，由候选 chunk 那次
	// visit 自己的角枚举处理）。块内格记账时夹回 range，防止把区间外的实心格记进来。
	bool block_sample = false;
	const int *blk_ir = nullptr;
	if (has_range[iterate_side])
	{
		blk_ir = range[iterate_side];
		const long long blk_vol = (long long)(blk_ir[3] - blk_ir[0] + 1)
			* (long long)(blk_ir[4] - blk_ir[1] + 1) * (long long)(blk_ir[5] - blk_ir[2] + 1);
		block_sample = blk_vol >= sBlockSampleMinVolume;
	}

	// ★ 块级记账已整体搬进 run_cell 的 block4 分支（按 4³ 块推进、记带标记的块号）。
	//   这里恒 0 ⇒ sRecordBlockAgg 关掉时的旧逐格路径逐位不变，不会被 "格号对齐" 污染。
	const int agg_mask = 0;

	// ★★★ 2026-11-07：挖洞记账 与 接触点/法线 **彻底分成两条通路**。
	//   true ⇒ 记账由「独立 4³ 块通路」在遍历之前一次跑完（按 4³ 块推进、块内零逐格），
	//          接触循环里不再有任何记账代码，退出条件只看接触配额；
	//   false ⇒ 旧的「逐格记账混在 run_cell 里」路径，供单测 A/B 做逐位基线。
	//   ★★ 不看 block_sample（2026-11-07 晚修正）：门槛 1024 是**接触几何**的分档，
	//     现场是 16³ 叶子 compound 互撞，单对重叠区普遍只有几百格 ⇒ block_sample 恒 false
	//     ⇒ 上一版 4³ 通路一次都没启用，日志仍是「记录 7764~12335 格」的逐格记账。
	//     现在只要这一对有记账通道，挖洞就一律走 4³；接触仍按 block_sample 决定逐格 / 2³ 抽样，
	//     两条通路互不影响（用户令：「碰撞点还是逐格计算，挖洞全部走 4³ 推进」）。
	const bool carve_separate = inRecord != nullptr && has_range[iterate_side]
		&& sRecordBlockAgg && sRecordBlockShift == 2;

	// ---- 3) 两侧的公共常量 ----
	//
	// 两侧需要的量是对称的，只是角色不同（遍历侧 / 候选侧），所以放同一张表里。
	struct GridSide
	{
		const VoxelShape *mVoxel;
		const SubShapeIDCreator *mIDCreator;
		Vec3 mScale;					// 形状缩放
		Mat44 mTransform;				// 盒心变换（= 入参，调用方已 sToGrid，见函数头说明）
		Mat44 mToWorld;					// 单位盒（含格长）-> 世界（不含格心平移）
		float mRadius;					// 该侧格子在全局分离轴上的投影半径
		float mWholeRadius;				// 该侧**整网格**在全局分离轴上的投影半径（判断两侧尺寸是否相当用）
		Vec3 mInvScale;					// 1 / mScale
		float mInvVoxelSize;			// 1 / 格长
		Vec3 mOrigin;					// 候选侧：格心世界坐标的基点
		Vec3 mStep[3];					// 候选侧：格心世界坐标的步进向量（下标 +1 走一格）
		Vec3 mFaceAxis[3];				// 单位化的格轴（= 格子的 6 张格面的法线方向），sNormalMode == 2 用
									// ★ 每查询只算一次（3 次 sqrt），mode 2 的"纯整面"分支直接拿它当接触法线 ——
									//   这是"法线归一化合并后再算"的落点：不要退回成逐格对归一化。
	};
	GridSide side[2];
	for (int i = 0; i < 2; ++i)
	{
		const VoxelShape *voxel = i == 0? inVoxel1 : inVoxel2;
		Vec3 scale = i == 0? inScale1 : inScale2;
		Mat44 transform = i == 0? inCenterOfMassTransform1 : inCenterOfMassTransform2;	// 已是盒心变换（见函数头）
		GridSide &s = side[i];

		s.mVoxel = voxel;
		s.mIDCreator = i == 0? &inSubShapeIDCreator1 : &inSubShapeIDCreator2;
		s.mScale = scale;
		s.mTransform = transform;
		s.mToWorld = transform * Mat44::sScale(scale);

		// 投影半径：同一侧所有格子的旋转都一样（格变换只改平移列），每查询算一次就够
		s.mRadius = sGetProjectionRadius(transform, scale * voxel->mVoxelSize, axis_world);
		s.mWholeRadius = sGetProjectionRadius(transform, scale * (voxel->mVoxelSize * Vec3(float(voxel->GetSizeX()), float(voxel->GetSizeY()), float(voxel->GetSizeZ()))), axis_world);

		s.mInvScale = Vec3(1.0f / scale.GetX(), 1.0f / scale.GetY(), 1.0f / scale.GetZ());
		s.mInvVoxelSize = 1.0f / voxel->mVoxelSize;

		// 格心的世界坐标 = origin + (x+0.5)*step[0] + (y+0.5)*step[1] + (z+0.5)*step[2]
		//（GetVoxelCenter = (格下标+0.5)*格长 - half_extent，线性 ⇒ 拆成基点 + 三个步进向量。
		//  ★ 步进向量是**方向**，必须用 Multiply3x3（线性部分）—— 用点变换会把平移加进每一步）
		s.mOrigin = s.mToWorld * (-voxel->mHalfExtent);
		s.mStep[0] = s.mToWorld.Multiply3x3(Vec3(voxel->mVoxelSize, 0, 0));
		s.mStep[1] = s.mToWorld.Multiply3x3(Vec3(0, voxel->mVoxelSize, 0));
		s.mStep[2] = s.mToWorld.Multiply3x3(Vec3(0, 0, voxel->mVoxelSize));
		for (int k = 0; k < 3; ++k)
			s.mFaceAxis[k] = s.mStep[k].Normalized();
	}

	bool iterate_is_1 = iterate_side == 0;
	const GridSide &iter = side[iterate_side];			// 遍历侧：逐格枚举它
	const GridSide &cand = side[iterate_is_1? 1 : 0];	// 候选侧：被跨格枚举查询


	// 遍历侧的未缩放局部空间 -> 候选侧的未缩放局部空间
	// ★ 两边用的入参都是盒心变换（见函数头）：GetVoxelCenter 返回盒心系坐标，
	//   直接拿入参相乘即可；若误用未经 sToGrid 的质心变换，mCenterOfMass ≠ 0 时整条映射漂 R*scale*m。
	Mat44 iter_to_cand = iterate_is_1
		? inCenterOfMassTransform2.Inversed() * inCenterOfMassTransform1 * Mat44::sScale(inScale1)
		: inCenterOfMassTransform1.Inversed() * inCenterOfMassTransform2 * Mat44::sScale(inScale2);

	// 遍历侧格心 -> 候选侧"格坐标系"（cell c 占据 [c, c+1]）的**仿射分解**。
	//
	// ★ 原来每格写的是
	//       c = ((iter_to_cand * GetVoxelCenter(ix,iy,iz)) * cand.mInvScale + cand.mHalfExtent) * cand.mInvVoxelSize;
	//   那是一次 4x4 矩阵乘（16 乘 + 12 加）外加两次逐轴线性运算 —— 而 GetVoxelCenter 和后面这两步
	//   全都是**仿射**的，整条链可以合成成 基点 + 三个步进向量，每格只剩 3 次乘加：
	//       c = cell_origin + (ix + 0.5) * cell_step[0] + (iy + 0.5) * cell_step[1] + (iz + 0.5) * cell_step[2]
	//   这是最内层、每格都要算的量，实测是整条体素 vs 体素窄相位里最贵的一环（32^3 时约占 1/3）。
	//
	// ★ 步进向量是**方向**，必须用 Multiply3x3（线性部分）—— 用点变换（operator*）会把平移量
	//   加进每一步，格心会随下标线性漂移，圈到的候选区间全错。
	//
	// 逐轴折进 invScale / invVoxelSize：v ↦ (v * mInvScale + mHalfExtent) * mInvVoxelSize
	// 是逐轴独立的仿射（缩放 kScale = mInvScale * mInvVoxelSize，平移 kOffset = mHalfExtent * mInvVoxelSize），
	// 所以直接乘到基点/步进上即可，不必每格再做一遍。
	const Vec3 cand_k_scale = cand.mInvScale * cand.mInvVoxelSize;
	const Vec3 cand_k_offset = cand.mVoxel->mHalfExtent * cand.mInvVoxelSize;
	const float iter_vs = iter.mVoxel->mVoxelSize;
	const Vec3 iter_to_cand_origin = iter_to_cand * (-iter.mVoxel->mHalfExtent);

	Vec3 cell_origin = iter_to_cand_origin * cand_k_scale + cand_k_offset;
	Vec3 cell_step[3] =
	{
		iter_to_cand.Multiply3x3(Vec3(iter_vs, 0, 0)) * cand_k_scale,
		iter_to_cand.Multiply3x3(Vec3(0, iter_vs, 0)) * cand_k_scale,
		iter_to_cand.Multiply3x3(Vec3(0, 0, iter_vs)) * cand_k_scale,
	};

	// ---- 3b) 跨格枚举 + 球面回收 ----
	//
	// ★ 格对筛选 = **跨格枚举**（straddle）+ 球面回收，两步：
	//   第一步：把遍历侧的格心映射到候选侧的格坐标系，每轴取 floor(c - 0.5 - e) .. ceil(c + 0.5 + e) - 1。
	//   一个格子（区间长度恒为 1 格）每轴最多跨 2 格，加上沿全局轴的推测余量 e 每轴最多 3 格
	//   —— 替代旧版的 8 角点 AABB 变换 + ExpandBy + sGetCellRange（每格省 ~10 次矩阵乘）。
	//   第二步（见下 dist_limit_sq）：在候选里做一次"3 减 3 乘"的球心距比较，把集合收回到球面内。
	//
	// 几何解释：把格子近似成内切球（边角冗余可接受）时，"两球相交 ⟺ 每轴 |Δ心| < 半径和"
	// ⟺ 对方格盒与我方格盒正面积重叠 ⟺ 对方格心落在跨格集合里 —— 跨格集合**包含**全部球相交对，
	// 但比球宽（盒子的角，格心距可达 1.7 格）；旋转的网格里这些角格对的数量是球相交对的 ~8 倍，
	// 深度检查拦不住（斜轴时侧面邻居的 |Δ·轴| 也很小），必须用球心距收回来。
	//
	// 半径和用于深度公式；接触点用各自一侧的半径（point_iter / point_cand 里已经带上了），
	// 不再需要单独的角色半径。
	float radius_sum = side[0].mRadius + side[1].mRadius;

	// 两侧各自的格长（世界单位）：mode 2 死区/深度、mode 1 接触点半径用。
	const float cell_len_0 = side[0].mVoxel->mVoxelSize * side[0].mScale.Abs().ReduceMin();
	const float cell_len_1 = side[1].mVoxel->mVoxelSize * side[1].mScale.Abs().ReduceMin();
	const float min_cell_len = cell_len_0 < cell_len_1? cell_len_0 : cell_len_1;

	// 格对球近似阈值（候选侧格坐标系单位）：格心距超过阈值（含推测余量）的格对没有接触。
	// ★ 半径用**轴对齐内切半格**（0.5 * 格长 * 最小缩放分量），不能用上面的投影半径 radius_*：
	//   斜轴下投影半径被旋转放大（0.5 格 -> ~0.61 格），球半径从 1.2 格涨到 ~1.4 格，
	//   接触数翻倍、查询慢 3 倍（实测）。内切半格是保守下界 ⇒ 球通过的格对必然通过深度检查
	//   （投影半径 >= 内切半格对任意轴成立），不会漏。
	// ★ 跨格枚举给的是"盒子重叠"集合（每轴 |Δ| < 1），比球近似宽 —— 网格带旋转时，盒子的角上
	//   会多出大量格心距 1 ~ 1.7 格的格对，而深度检查拦不住它们（斜轴时侧面邻居的 |Δ·轴| 也很小），
	//   实测接触数 ~8 倍、查询 5 倍慢。所以跨格之后还要用一次"3 减 3 乘 + 比较"把集合收回到球面内
	//   —— 这正是"把体素当球"的语义本身，不是可省的步骤。
	float filter_dist = 0.5f * (inVoxel1->mVoxelSize * inScale1.Abs().ReduceMin()
							+ inVoxel2->mVoxelSize * inScale2.Abs().ReduceMin());
	float dist_limit = filter_dist / (cand.mVoxel->mVoxelSize * cand.mScale.Abs().ReduceMin());

	// ★ B 方案块抽样的派生常量（逐格路径全为旧值 ⇒ 逐位一致）：
	//   blk_half_cand = 遍历侧"块"的半宽（候选格单位）。逐格 = 半格 0.5；块 = 2 格宽 ⇒ 半宽 =
	//   一个遍历格长折算到候选格单位（cell_len_iter / cell_len_cand）。
	//   half_sum4：mode 2 深度公式的"两半宽之和"（候选格单位）—— 逐格 1.0（半格 + 半格），
	//   块 = blk_half_cand + 0.5。深度 = min_cell_len * (half_sum4 − dmax4)。
	//   filter_lim：球筛半径（候选格单位）= 遍历侧内切半宽 + 候选侧内切半格；块多出一个遍历半格。
	//   mode 0 / mode 1 全局轴分支的 radius_sum、mode 1 球分支的 filter_dist 同理各加一份遍历半宽。
	const float cand_cell_len = cand.mVoxel->mVoxelSize * cand.mScale.Abs().ReduceMin();
	const float iter_cell_len = iterate_is_1? cell_len_0 : cell_len_1;
	const float blk_half_cand = iter_cell_len / cand_cell_len;
	const float half_sum4 = block_sample? (blk_half_cand + 0.5f) : 1.0f;
	const float filter_lim = block_sample? dist_limit + 0.5f * blk_half_cand : dist_limit;
	const float filter_lim_sq = filter_lim * filter_lim;
	const float radius_sum_sel = block_sample? radius_sum + iter.mRadius : radius_sum;
	const float filter_dist_sel = block_sample? filter_dist + 0.5f * iter_cell_len : filter_dist;

	// ---- 4) 遍历**遍历侧**的实心格；每一格用跨格枚举找出**候选侧**里的候选实心格 ----
	//
	// 遍历顺序用 <true>：**由外往中心**。接触数撞到 sMaxContactPoints 截断时，先被保留的是重叠区
	// 外圈的格子 —— 剩下的接触仍然铺满整片区域的外围（形心和整片重合），不会像顺序截断那样只剩
	// 一侧（那会让物体被支撑在单边，发生倾斜/下陷，正是最早那个"下陷又被推上来"的成因链）。
	int num_contacts = 0;

	// ★★★ 配额打满之后还要不要继续走：**只有还需要记账**才继续（撞击记账要的是这一击撞到的
	//   完整格集合，见 sMaxContactPoints 的说明；两侧记账都满 ⇒ 再扫也不会记，真的停）。
	//   ★ 以前外层一律按 `num_contacts < sMaxContactPoints` 停 ⇒ 配额一满就再也不调用 run_cell，
	//     run_cell 里的 keep_scanning / can_record 根本走不到，记账在配额处被硬截断。
	//     现在配额满之后的行走只花"每格一次块级粗判 + 整块记账"（run_cell 的 try_record_block），
	//     不再逐格产接触，所以可以放心扫完重叠区。
	auto still_walking = [&]() -> bool
	{
		if (num_contacts < sMaxContactPoints)
			return true;
		if (carve_separate)
			return false;								// ★ 记账已独立跑完 ⇒ 接触配额满就真停，不再为记账空转
		return inRecord != nullptr
			&& (inRecord->mCount1 < uint(VoxelShape::sMaxCellsPerSide)
			 || inRecord->mCount2 < uint(VoxelShape::sMaxCellsPerSide));
	};

	// 候选侧的常量（不随遍历格变）——提到 lambda 外面，省掉每格重复的取分量 + 数组构造
	const int size[3] = { cand.mVoxel->mSizeX, cand.mVoxel->mSizeY, cand.mVoxel->mSizeZ };
	const int smin[3] = { cand.mVoxel->GetSolidMin(0), cand.mVoxel->GetSolidMin(1), cand.mVoxel->GetSolidMin(2) };
	const int smax[3] = { cand.mVoxel->GetSolidMax(0), cand.mVoxel->GetSolidMax(1), cand.mVoxel->GetSolidMax(2) };

	// ---- 3c) 2x2x2 块掩码（mode 2 用）----
	//
	// 块 = 每轴取 lo 或 lo+1 的 8 格，位序 bit = i | (j << 1) | (k << 2)（i/j/k = x/y/z 方向偏移），
	// 与 sNbPlans / sEmptyPlans 的掩码位序完全一致。mode 2 的"分类"读它。
	//
	// ★ 2x2x2 只跨 1~8 个 4x4x4 子块：每轴两个坐标落在同一组（低 2 位都 != 3）时该轴不产生新字
	//   —— 84% 的格对只跨 1~2 个字。这里按字缓存：遍历顺序 i（x）最快，只有 x 跨组时才重读，
	//   平均 1.95 次而不是 8 次。位序 = (x&3) | (y&3)<<2 | (z&3)<<4，与 IsSolid 逐位一致。
	//   边界：先整体预检一次（2x2x2 全在格内），越界回退到逐格 IsSolid（越界 = 空）。
	auto build_block_mask = [](const VoxelShape *inShape, int inLo0, int inLo1, int inLo2) -> int
	{
		const int lo[3] = { inLo0, inLo1, inLo2 };
		int mask = 0;
		if (lo[0] >= 0 && lo[1] >= 0 && lo[2] >= 0
			&& lo[0] + 1 < inShape->mSizeX && lo[1] + 1 < inShape->mSizeY && lo[2] + 1 < inShape->mSizeZ)
		{
			const uint64 *occ = inShape->mOccupancy;
			const int tx = inShape->mTexelsX;
			const int tp = tx * inShape->mTexelsY;
			const int wx[2] = { lo[0] >> cSubChunkShift, (lo[0] + 1) >> cSubChunkShift };
			const int wy[2] = { lo[1] >> cSubChunkShift, (lo[1] + 1) >> cSubChunkShift };
			const int wz[2] = { lo[2] >> cSubChunkShift, (lo[2] + 1) >> cSubChunkShift };
			const int bx[2] = { lo[0] & 3, (lo[0] + 1) & 3 };
			const int by[2] = { lo[1] & 3, (lo[1] + 1) & 3 };
			const int bz[2] = { lo[2] & 3, (lo[2] + 1) & 3 };
			// 平铺到一维下标的三个分量提到循环外（每个字的下标 = pw2[k] + pw1[j] + wx[i]）
			const int pw1[2] = { wy[0] * tx, wy[1] * tx };
			const int pw2[2] = { wz[0] * tp, wz[1] * tp };

			int last = -1;
			uint64 word = 0;
			for (int k = 0; k < 2; ++k)
				for (int j = 0; j < 2; ++j)
					for (int i = 0; i < 2; ++i)
					{
						const int idx = pw2[k] + pw1[j] + wx[i];
						if (idx != last)
						{
							word = occ[idx];
							last = idx;
						}
						if ((word >> (bx[i] | (by[j] << 2) | (bz[k] << 4))) & 1)
							mask |= 1 << (i | (j << 1) | (k << 2));
					}
		}
		else
		{
			for (int k = 0; k < 2; ++k)
				for (int j = 0; j < 2; ++j)
					for (int i = 0; i < 2; ++i)
						if (inShape->IsSolid(lo[0] + i, lo[1] + j, lo[2] + k))
							mask |= 1 << (i | (j << 1) | (k << 2));
		}
		return mask;
	};

	// ---- 4a) 一个遍历格的完整处理：候选侧格坐标 c、候选区间 lo/hi 已由调用方算好传进来 ----
	//
	// mode 2 块掩码缓存：块起点 lo4[k] 只随**候选格 a** 变（= a[k] 或 a[k]-1），相邻遍历格的
	// 候选格常落在同一个块上 ⇒ 记下上一次的起点与掩码，命中就跳过 build_block_mask。
	// 与"窗口起点 floor(c-0.5)"的关系见 VoxelShape.h 的 sBlockMaskCache 说明。
	// ★★ 2026-11-06：缓存**放在 run_cell 外面**（原来声明在 run_cell 内 ⇒ 每次调用都重新建，
	//   而严格对齐时每个 run_cell 只处理 1 个候选格，等于每格白付 8 次 IsSolid 建掩码）。
	//   遍历沿列推进、候选格每轴每 2 格才换一次块 ⇒ 跨格命中率约 50%。位图在一次查询内不变，
	//   mode 2 与 mode 1 的 neighborhood_full8 都只查**候选侧**（同一个 cand.mVoxel）⇒ 共用安全。
	int cache_lo4[3] = { 0, 0, 0 };
	int cache_mask4 = 0;
	bool cache_valid = false;

	// 为什么把前置量拆出来：**每格的固定开销**（格心 → c → 候选区间）跟这一格有没有候选、有几个
	// 候选都无关，是纯"每格一次"的成本；而候选循环平均每格只跑 1.3~1.8 次（实测）。所以这条路径
	// 的瓶颈是每格的固定开销，不是候选循环 —— 把 4 个遍历格的固定开销一起算（见 4b）才有意义。
	// 候选循环本身是数据相关的、长度不定，仍然逐格标量跑。
	auto run_cell = [&](int inX, int inY, int inZ, Vec3Arg inCenterIter, const float (&inCC)[3], const int (&inLo)[3], const int (&inHi)[3])
	{
		const bool hot = sHotStatsEnabled.load(std::memory_order_relaxed);
		if (hot)
			sStatSolidCells.fetch_add(1, std::memory_order_relaxed);		// 调试统计：实心、进入接触计算的格
		// 遍历侧这一格的 SubShapeID：**懒算** —— 只有真正产出接触的格对才需要
		// （静止球 4990 个实心格只产出 ~11 个接触，逐格预先编码是纯浪费）。
		SubShapeID id_iter;
		bool id_iter_done = false;

		// sInteriorSkipPairs：这一格是不是内部格。同样**懒算**（第一次用到时算一次）。
		bool iter_is_interior = false;
		bool iter_interior_done = false;

		// ★ 承压按格去重（遍历侧）：这一格的格号（inX/inY/inZ）在本轮是**常量**，只需一个 bool
		//   记住"这一格的强度已经计过了"。候选侧没有这个便利，走 sDedupFirstHit 查表。
		bool iter_recorded = false;

		// ★★★ B 方案块抽样（block_sample）专有状态：
		//   contact_done：本块已经产出过 1 个代表接触 ⇒ 窗口里剩下的候选格只记账、不再算几何。
		//   blk_recorded / record_block_cells()：块的**遍历侧 8 格**记账。要等窗口里出现第一颗
		//     实心候选格才记（窗口全空 = 这个块根本没碰到对方 ⇒ 一格都不许记，否则会把没撞到
		//     的格整块挖掉）。8 格互不重复（块角唯一），不用 dedup；格号夹回 blk_ir（range），
		//     防止块伸出重叠区间时把区间外的实心格记进来。记完置 iter_recorded ⇒ 下面
		//     record_pair 的遍历侧分支自动停用，候选侧照常去重记账。
		bool contact_done = false;
		bool blk_recorded = false;
		auto record_block_cells = [&]()
		{
			if (blk_recorded || inRecord == nullptr)
				return;
			blk_recorded = true;
			iter_recorded = true;
			if (agg_mask != 0)
			{
				// ★ 记账聚合（A/B）：遍历侧只记 **1 条**（块角格号），不再循环块内 8 格。
				const int bx = inX & ~agg_mask, by = inY & ~agg_mask, bz = inZ & ~agg_mask;
				if (bx >= blk_ir[0] && by >= blk_ir[1] && bz >= blk_ir[2]
					&& bx <= blk_ir[3] && by <= blk_ir[4] && bz <= blk_ir[5])
				{
					const float strength = iter.mVoxel->GetContactStrengthAtCell(bx, by, bz);
					const uint32_t cell = sPackCell(uint(bx), uint(by), uint(bz));
					if (iterate_is_1)
					{
						inRecord->mStrength1 += strength;
						sRecordCell(inRecord, true, cell);
					}
					else
					{
						inRecord->mStrength2 += strength;
						sRecordCell(inRecord, false, cell);
					}
				}
				return;
			}
			for (int dz = 0; dz < 2; ++dz)
				for (int dy = 0; dy < 2; ++dy)
					for (int dx = 0; dx < 2; ++dx)
					{
						const int bx = inX + dx, by = inY + dy, bz = inZ + dz;
						if (bx < blk_ir[0] || by < blk_ir[1] || bz < blk_ir[2]
							|| bx > blk_ir[3] || by > blk_ir[4] || bz > blk_ir[5])
							continue;					// 块伸出重叠区间的部分不记
						if (!iter.mVoxel->IsSolid(bx, by, bz))
							continue;
						const float strength = iter.mVoxel->GetContactStrengthAtCell(bx, by, bz);
						const uint32_t cell = sPackCell(uint(bx), uint(by), uint(bz));
						if (iterate_is_1)
						{
							inRecord->mStrength1 += strength;
							sRecordCell(inRecord, true, cell);
						}
						else
						{
							inRecord->mStrength2 += strength;
							sRecordCell(inRecord, false, cell);
						}
					}
		};

		// ★ 2x2x2 邻域「八格全实心」判定（= 这一格被自己一侧的实心整格埋住）。
		//   mode 1 用它把**深埋**格对切到整网格全局轴：逐对格心连线在深嵌时方向互相抵消
		//   （有的格对往上推、有的往下推）⇒ 净推进力为 0 ⇒ 方块卡在地形里顶不出来。
		//   邻域起点公式、位图构建与缓存都和 mode 2 的第 1) 步完全一致（共用同一份块掩码缓存）。
		auto neighborhood_full8 = [&](const VoxelShape *inShape, int inX2, int inY2, int inZ2, const float (&inCC2)[3]) -> bool
		{
			const int ac2[3] = { inX2, inY2, inZ2 };
			int lo2[3];
			for (int k = 0; k < 3; ++k)
				lo2[k] = inCC2[k] >= float(ac2[k]) + 0.5f? ac2[k] : ac2[k] - 1;

			int mask;
			if (sBlockMaskCache != 0 && cache_valid
				&& lo2[0] == cache_lo4[0] && lo2[1] == cache_lo4[1] && lo2[2] == cache_lo4[2])
				mask = cache_mask4;
			else
			{
				mask = build_block_mask(inShape, lo2[0], lo2[1], lo2[2]);
				if (sBlockMaskCache != 0)
				{
					cache_valid = true;
					cache_lo4[0] = lo2[0];
					cache_lo4[1] = lo2[1];
					cache_lo4[2] = lo2[2];
					cache_mask4 = mask;
				}
			}

			// 8 位 popcount（不引入 <bit>）
			int n = mask - ((mask >> 1) & 0x55);
			n = (n & 0x33) + ((n >> 2) & 0x33);
			n = (n + (n >> 4)) & 0x0F;
			return n == 8;
		};


		// ★★★ 撞击破坏第 1 步的记账动作：记录**这一对相交体素**（格号 + 该格材质承压）。
		//   参数是候选侧格号，遍历侧用 inX/inY/inZ。两侧按 shape1/shape2 角色分配
		//   （与接触点同一约定），上层按墙侧读。
		//   ★ 只做「查该格材质强度 + 存格号」：不编码 SubShapeID ⇒ 调用点可以放在
		//     相交判定的那一刻（法线/深度之前）。接触点配额只截接触不截记账（2026-11-05 改）：
		//     配额满后的格对退回第一层块级粗记账（try_record_block），重叠格集合仍然记全。
		auto record_pair = [&](int x, int y, int z)
		{
			if (inRecord == nullptr)
				return;

			// ★ 记账聚合（A/B，默认关）：候选侧格号对齐到块角 ⇒ 同一块被几十颗候选格命中
			//   也只记 1 条（去重走现有 sDedupFirstHit，key 换成块角格号）。
			const int agg_m = agg_mask;

			// ★★ 按格去重（见 sDedupFirstHit 的说明）：一颗格只计一次强度、只记一次格号。
			//   遍历侧 = 本轮常量（bool）；候选侧 = 被相邻遍历格重复枚举（哈希）。
			const uint32_t cell_iter = sPackCell(uint(inX), uint(inY), uint(inZ));
			const uint32_t cell_cand = agg_m == 0? sPackCell(uint(x), uint(y), uint(z))
				: sPackCell(uint(x & ~agg_m), uint(y & ~agg_m), uint(z & ~agg_m));

			// ---- shape1 侧 ----
			bool first1 = iterate_is_1? !iter_recorded : sDedupFirstHit(inRecord->mDedupGen, cell_cand);
			if (iterate_is_1)
				iter_recorded = true;
			if (first1)
			{
				const VoxelShape *v1 = iterate_is_1? iter.mVoxel : cand.mVoxel;
				int c1x = iterate_is_1? inX : x;
				int c1y = iterate_is_1? inY : y;
				int c1z = iterate_is_1? inZ : z;
				inRecord->mStrength1 += v1->GetContactStrengthAtCell(c1x, c1y, c1z);
				sRecordCell(inRecord, true, iterate_is_1? cell_iter : cell_cand);
			}

			// ---- shape2 侧 ----
			bool first2 = iterate_is_1? sDedupFirstHit(inRecord->mDedupGen, cell_cand) : !iter_recorded;
			if (!iterate_is_1)
				iter_recorded = true;
			if (first2)
			{
				const VoxelShape *v2 = iterate_is_1? cand.mVoxel : iter.mVoxel;
				int c2x = iterate_is_1? x : inX;
				int c2y = iterate_is_1? y : inY;
				int c2z = iterate_is_1? z : inZ;
				inRecord->mStrength2 += v2->GetContactStrengthAtCell(c2x, c2y, c2z);
				sRecordCell(inRecord, false, iterate_is_1? cell_cand : cell_iter);
			}
		};
		// ---- 3d) 第一层：把"遍历格 + 它的候选窗口"整块当一颗直径 2 格的体素粗判 ----
		//
		// ★★★ 接触点配额（sMaxContactPoints）打满之后的专用路径：
		//   大面积重叠时一帧能撞上万格，而配额满了以后一个接触点也产不出来，剩下的唯一任务是
		//   "把这一击撞到的格集合记下来"。此时**第二层（下面那个逐格循环：IsSolid + 球筛 +
		//   内部格判定 + 材质查表 + 法线/深度/接触点）整个不跑**，改在这里做一次块级粗判：
		//   把遍历格当成"直径 2 格的体素"，以它的盒心（= inCC，已经是候选侧格坐标）按单格
		//   体素的**同一套 straddle** 取候选侧的 2x2x2 盒范围（= inLo..inHi 本身），判据就只有
		//   "这 8 格里有没有实心格"——**不算距离、不碰顶点、不做矩阵运算**。有就把这 8 格直接
		//   记账（承压只累加第一颗实心格，不为这 8 格各查一次材质表、也不为记账再回头重扫）。
		//   代价（刻意接受）：两侧盒心最多差半格、盒又是按内切取的，边缘那一列格可能漏记；
		//   承压和也只有实际命中格的一小部分 ⇒ 破坏判据更容易过。只在"接触点已经打满"的极端重叠里生效。
		// ★ 返回 true = 本窗口已经处理完（不是"没重叠"）⇒ 调用点直接 return，第二层不再跑。
		auto try_record_block = [&]() -> bool
		{
			// ★ 分离模式：挖洞记账已由独立 4³ 通路一次跑完，任何逐格粗记账都不许再写，
			//   否则记录里混进普通格号 ⇒ 下游又退回逐格挖（现场「记录上万格」的直接来源）。
			if (carve_separate)
				return false;
			if (inRecord == nullptr || num_contacts < sMaxContactPoints)
				return false;							// 配额没满：照常跑第二层，逐格产接触
			// 记账两侧都满了 ⇒ 这一格什么都不用做。
			// ★ 这里**现读计数**（不用上面那份 can_record 快照）：记账是跨调用累积的，配额打满之后
			//   这条路径每格都会被调用，快照会把"已经记满"的 pair 一直当没满。
			if (inRecord->mCount1 >= uint(VoxelShape::sMaxCellsPerSide)
				&& inRecord->mCount2 >= uint(VoxelShape::sMaxCellsPerSide))
				return true;

			// ---- 第一层：粗判与记账**同一趟**扫完（最多 8 格） ----
			// 遍历格当成一颗"直径 2 格的体素"（半长 1 格），它的盒心就是遍历格心 inCC —— 而 inCC
			// 已经是候选侧格坐标（第 3 步的格心变换算出来的），"把盒心转到对方局部空间"这句不用再做。
			// 以这个中心点走**和单格体素完全同一套 straddle**（中心 ± 半格 + 夹取）得到候选侧的盒范围，
			// 那就是传进来的 inLo..inHi 本身（≤ 2x2x2 = 8 格）：两边都是 2x2x2 盒，粗判只剩一句
			// "这 8 格里有没有实心格"。
			// ★ 不取盒心、不算距离、不碰 8 个顶点、不做矩阵运算 —— 全部复用已经算好的量，零额外开销。
			// ★ 两侧盒心最多差半格，边缘那一列格可能漏记（粗判，接受）；漏的方向是"少记"，不会多记。
			// ★ 扫完就 return true：这 8 格空了第二层也不会有东西，且不再回头重扫一遍。
			bool any_solid = false;
			bool strength_done = false;
			for (int k = inLo[2]; k <= inHi[2]; ++k)
				for (int j = inLo[1]; j <= inHi[1]; ++j)
					for (int i = inLo[0]; i <= inHi[0]; ++i)
					{
						if (!cand.mVoxel->IsSolid(i, j, k))
							continue;					// 空格不进记录，省得把格号容量烧在空气上

						// 第一颗实心 ⇒ 这一击确实相交：遍历侧（本轮常量）格号 + 承压在此记一次
						if (!any_solid)
						{
							any_solid = true;
							if (!iter_recorded)
							{
								const uint32_t cell_iter0 = sPackCell(uint(inX), uint(inY), uint(inZ));
								if (iterate_is_1 || sDedupFirstHit(inRecord->mDedupGen, cell_iter0))
								{
									iter_recorded = true;
									const float strength_iter = iter.mVoxel->GetContactStrengthAtCell(inX, inY, inZ);
									if (iterate_is_1)
									{
										inRecord->mStrength1 += strength_iter;
										sRecordCell(inRecord, true, cell_iter0);
									}
									else
									{
										inRecord->mStrength2 += strength_iter;
										sRecordCell(inRecord, false, cell_iter0);
									}
								}
							}
						}

						// 候选侧：这一格的格号
						const uint32_t cell_cand = sPackCell(uint(i), uint(j), uint(k));
						if (!sDedupFirstHit(inRecord->mDedupGen, cell_cand))
							continue;					// 已被相邻遍历格的窗口枚举过
						// 承压只取**第一颗**（同一趟顺手取，不为这 8 格各查一次材质表）
						if (!strength_done)
						{
							strength_done = true;
							const float strength_cand = cand.mVoxel->GetContactStrengthAtCell(i, j, k);
							if (iterate_is_1)
								inRecord->mStrength2 += strength_cand;
							else
								inRecord->mStrength1 += strength_cand;
						}
						sRecordCell(inRecord, iterate_is_1? false : true, cell_cand);
					}
			return true;
		};

		// ★★★ 第一层先跑一次：接触点配额已经打满（大面积重叠的主路段）⇒ 第二层整个不跑，
		//   只做一次块级粗判 + 整块记账。（放在计时段之前，避免提前 return 漏记探测耗时。）
		//   ★ 块抽样路径不走这里：块的记账（record_block_cells + 窗口逐格 record_pair）和
		//     每块一个代表接触都由下面的第二层循环一手包办，这里再扫一遍是重复劳动。
		if (!block_sample && !carve_separate && try_record_block())
			return;

		// ---- 4) 逐格枚举：遍历侧的实心格 × 跨格枚举出的候选实心格 ----
		// 诊断：候选循环累计耗时（含其中的 IsSolid/球面筛选/死区预判/分类），随最大调用快照带出。
		// 时钟读取是逐格纯开销（满载时放大数倍），关热点诊断时跳过。
		std::chrono::steady_clock::time_point probe_t0;
		const bool probe_timed = hot;
		if (probe_timed)
			probe_t0 = std::chrono::steady_clock::now();
		// ★★★ 遍历上限与「接触点上限」**解耦**（2026-10-29）：
		//   记账路径（inRecord != nullptr，即 body pair 的撞击窄相位）要的是**这一击撞到的完整格集合**，
		//   接触点只是给求解器的一批采样 ⇒ 后者到顶时跳过"产接触"，但记账必须扫完全部重叠格。
		//   旧行为是"接触点满 ⇒ 三层循环整体停" ⇒ 高速深嵌时重叠上万的格里只记到前 N 个，
		//   洞挖不干净 ⇒ 下一帧重叠更多 ⇒ 越撞越穿（用户假设，与实测"记录 4096 打满"吻合）。
		//   非记账路径（CharacterVirtual 的脚下查询、ShapeCast 等）inRecord 为 null ⇒ 行为完全不变。
		//   记录满两侧上限时也停（再扫也不会记）。
		//   ★ 注意"记录未满"要**在扫描前取快照**：记录是跨调用累积的（同一形状对多次碰撞复用同一条），
		//     拿累积值当条件会让"已经记满的 pair"一次循环都不进 ⇒ 一个接触都不产（实测过）。
		//   ★★ 2026-11-05：外层的行走条件也从 `num_contacts < sMaxContactPoints` 改成了
		//     still_walking() —— 否则配额一满就不再调用 run_cell，上面这套"配额满只记账"的分支
		//     一次都走不到（记账在配额处被硬截断）。配额满之后的每格只做第一层的块级粗判 + 整块
		//     记账（try_record_block），扫完整个重叠区的代价 = "每格 ≤ 8 次 IsSolid 位测试"。
		const bool can_record = inRecord != nullptr
			&& (inRecord->mCount1 < uint(VoxelShape::sMaxCellsPerSide)
			 || inRecord->mCount2 < uint(VoxelShape::sMaxCellsPerSide));
		auto keep_scanning = [&]() -> bool
		{
			if (num_contacts < sMaxContactPoints)
				return true;					// 还在产接触：照常扫
			return !carve_separate && can_record;	// ★ 分离模式下记账已独立完成，配额满就停
		};

		// ★★★ B 方案：本块的**代表格**（块模式下至多这一格产接触）。
		//   定义 = 块心（inCC）所在的候选格；是空格就沿**同一列**（x/z 不变）上下找最近的实心格。
		//   ★ 为什么不能用"窗口扫描第一个通过者"：分块地形里同一个块会被多个叶子查询各扫一次，
		//     每次的窗口（夹到各自叶子的实心范围）不一样 ⇒ "第一个"随查询切法变 ⇒ 单网格 vs 复合体
		//     接触数对不上（VoxelShapeCompoundTerrain 的逐位一致检查当场红）。
		//   ★ 为什么螺旋找最近实心格也不行：复合路径里每片叶子只看得见自己的格，接缝两侧的叶子
		//     会给同一个块各找一个"最近实心格" ⇒ 一个块产出 2~4 条重复接触（实测 14 vs 89）。
		//     **列**是唯一归属一片叶子的（x/z 定死）⇒ 拥有这一列的叶子找到的格 = 单网格会找到的
		//     同一颗格，别的叶子在这一列上无格可找 ⇒ 天然去重、两方案逐位一致。
		//   ★ 含入格在重叠场景必然实心（块心扎在对方实体里）⇒ 列搜索只是擦边/浅重叠的兜底。
		int rep_c[3] = { 0, 0, 0 };
		bool rep_valid = false;
		if (block_sample)
		{
			const int c0[3] = { int(std::floor(inCC[0])), int(std::floor(inCC[1])), int(std::floor(inCC[2])) };
			if (cand.mVoxel->IsSolid(c0[0], c0[1], c0[2]))
			{
				rep_c[0] = c0[0]; rep_c[1] = c0[1]; rep_c[2] = c0[2];
				rep_valid = true;
			}
			// 列搜索余量 = 块半宽 + 1（原来的 2 格块是 ±2，与 B 方案一致）
			else for (int dy = -2; dy <= 2 && !rep_valid; ++dy)
			{
				if (dy == 0)
					continue;
				if (cand.mVoxel->IsSolid(c0[0], c0[1] + dy, c0[2]))
				{
					rep_c[0] = c0[0]; rep_c[1] = c0[1] + dy; rep_c[2] = c0[2];
					rep_valid = true;
				}
			}
		}

		// ★★★ 2026-11-07：挖洞记账已**整体搬出** run_cell，改由遍历之前的独立 4³ 块通路一次跑完。
		//   这里只剩「接触几何」：块抽样时把三重循环缩到代表格 rep_c 一格，
		//   法线 / 深度 / 接触位置 / SubShapeID 全在原路径里算，一行未改。
		//   ★ 无记账通道（CharacterVirtual 脚下、ShapeCast）的大重叠查询同样缩到 rep_c，
		//     这是上一轮的性能修正，与记账无关，保持不变。
		const bool block4 = block_sample && sRecordBlockAgg && sRecordBlockShift == 2;
		int lo_run[3] = { inLo[0], inLo[1], inLo[2] };
		int hi_run[3] = { inHi[0], inHi[1], inHi[2] };
		if (block4)
		{
			// 接触配额已满 ⇒ 后面几何不必再跑（记账不受影响，它已独立跑完）
			if (num_contacts >= sMaxContactPoints)
				return;
			// 循环缩成 rep_c 一格：接触仍由同一颗代表格按原路径产出
			if (rep_valid)
			{
				for (int k = 0; k < 3; ++k) { lo_run[k] = rep_c[k]; hi_run[k] = rep_c[k]; }
			}
			else
			{
				lo_run[0] = 1; hi_run[0] = 0;		// 空窗口，不产接触
			}
		}
		for (int z = lo_run[2]; z <= hi_run[2] && keep_scanning(); ++z)
			for (int y = lo_run[1]; y <= hi_run[1] && keep_scanning(); ++y)
				for (int x = lo_run[0]; x <= hi_run[0] && keep_scanning(); ++x)
				{
				if (hot)
					sStatProbes.fetch_add(1, std::memory_order_relaxed);	// 调试统计：格对探测次数
					if (!cand.mVoxel->IsSolid(x, y, z))
						continue;

					// 球近似筛选（候选侧格坐标系单位）：跨格集合比球宽（角上的格对），收回到球面内。
					// 直接用格下标算，不需要先算格心的世界坐标 —— 被刷掉的格对连 center_cand 都不用算。
					// ★ 块抽样：遍历侧当"直径 2 格的体素"，球心 = 块心（inCC 传进来的就是块心）、
					//   半径多出一个遍历半格（filter_lim_sq 在上面按 block_sample 选好）。
					float dx = float(x) + 0.5f - inCC[0];
					float dy = float(y) + 0.5f - inCC[1];
					float dz = float(z) + 0.5f - inCC[2];
				if (dx * dx + dy * dy + dz * dz > filter_lim_sq)
					continue;

				// ★ mode 2（2x2x2 邻域分类）廉价预判：接触存活的**充要**条件就是后面死区检查的那一条
				//   dmax = max(|dx|,|dy|,|dz|) < 0.95（depth = cell*(1-dmax) > cell*0.05）。
				//   dx/dy/dz 上面已经算好，这里提前用同一条判据毙掉注定无接触的格对，
				//   跳过内部格检查、center_cand 和整个 mode 2 分类（Length/归一化/查表）。
				//   判据写法与死区处逐字一致 ⇒ 逐位等价：被毙的格对本来就不会产出接触。
				float dmax4 = 0.0f;
				if (sNormalMode == 2)
				{
					dmax4 = max(abs(dx), max(abs(dy), abs(dz)));
					// ★ 块抽样：half_sum4 = 块半宽 + 候选半格（逐格路径 = 1.0，逐位一致）
					if (min_cell_len * (half_sum4 - dmax4) <= min_cell_len * 0.05f)
						continue;
				}

				// ★★★ B 方案块抽样：候选格过了球筛（+ mode 2 死区）⇒ 这一对是**真重叠**，才记账。
				//   ★★ 记账必须放在筛选**之后**（2026-11-06 实测修正）：一开始放在 IsSolid 之后
				//   —— 深穿透时每块窗口 30~60 个实心格全记（哈希 + 材质查表各一次），
				//   单查询 23 万次 record_pair，比逐格路径慢 15 倍。放在这里后记账量回到与
				//   逐格路径同一量级（逐格路径也是"过了球筛 + 死区才记"）。
				//   记账范围：块的遍历侧 8 格（夹回 range）+ 这颗候选格（dedup 去重、承压一颗一次）。
				//   接触：只有**代表格**（块心所在格，见上面的 rep_c）往下走几何；其余候选格只记账。
			if (block_sample && !block4 && !carve_separate)
			{
				record_block_cells();
				record_pair(x, y, z);
			}
			if (block_sample)
			{
				if (contact_done || num_contacts >= sMaxContactPoints)
					continue;
				if (x != rep_c[0] || y != rep_c[1] || z != rep_c[2])
					continue;					// 非代表格：只记账，不算几何
			}

					// 内部格过滤（sInteriorSkipPairs）：1 = 两侧都内部才跳；2 = 任一侧内部就跳。
						// 内部格 = 6 邻全实心、埋在自己实体深处，它的接触与表面格的接触重复/冗余。
						// ★★ 跳的只是**接触**（含下面的几何），不是记账（2026-11-05 用户令）：
						//   "只要重叠区域都不能跳过，必须要记账" —— 深穿透时中间那片全是内部格 x 内部格
						//   的格对，整片跳掉 = 整片不记 = 球体中间那段永远削不掉、最后被求解器顶出来。
						//   这里在 continue 之前先把这对格记下来（球筛已过 ⇒ 两格的球已重叠，
						//   mode 2 更是已过死区 = 确定相交；mode 0/1/4 可能多记一层亚格级的擦边对，
						//   宁可多记不可漏记）。接触集合不变：跳掉的对本来就不产接触。
					if (sInteriorSkipPairs != 0)
					{
					if (!iter_interior_done)
					{
						iter_is_interior = iter.mVoxel->IsVoxelInterior(inX, inY, inZ);
						// ★ 块抽样：遍历侧按**整块 8 格**判内部 —— 块角埋在实体深处、但块里含着
						//   表面格（块面露在空气/接触面上）的块不能算内部。skip=2（任一侧内部就跳）
						//   时尤其致命：深埋方块 vs 基岩，接触界面全靠"遍历侧表面格 × 对方格"的对，
						//   按块角判内部会把含表面层的块整块判成内部 ⇒ 接触全部跳掉 ⇒ 方块穿底
						//   （VoxelShapeGlobalAxisSignBedrock 实测 y 从 0 掉到 -60）。
						//   ★ 只在块模式下做（块角才是角格）；逐格路径必须保持旧版单格判定逐位不变。
						if (block_sample)
							for (int k = 1; k < 8 && iter_is_interior; ++k)
								iter_is_interior = iter.mVoxel->IsVoxelInterior(inX + (k & 1), inY + ((k >> 1) & 1), inZ + ((k >> 2) & 1));
						iter_interior_done = true;
					}
						bool cand_is_interior = cand.mVoxel->IsVoxelInterior(x, y, z);
					if (sInteriorSkipPairs == 2? (iter_is_interior || cand_is_interior)
						: (iter_is_interior && cand_is_interior))
					{
						// ★ 块抽样：记账已在 IsSolid 处一手包办（更早、更全），这里只管跳接触。
						if (inRecord != nullptr && !carve_separate && !block_sample)
						{
							record_pair(x, y, z);
							// 配额刚打满 ⇒ 本窗口剩下的格退回块级粗记账（与 mode 2 分支同款）
							if (try_record_block())
								return;
						}
						continue;
					}
					}

					// ============================================================
					// ★★★ 撞击破坏第 1 步：**记录这一对相交体素**（格号 + 该格材质承压）
					//
					//   位置 = 相交判定的那一刻，不是后面产接触点的地方：mode 2 的相交充要条件
					//   就是上面那条死区检查（depth = r*(1-dmax) > 0.05r），过了它这两格确定相交，
					//   x/y/z 与 inX/inY/inZ 都在手上 ⇒ 当场记账，不碰法线、不碰 SubShapeID。
				//   ★ 接触点配额（sMaxContactPoints）只限制**接触**，不限制记账：满了之后的格对
				//     退回第一层块级粗记账（try_record_block），格集合仍然记全（2026-11-05 改）。
				//   ★ 块抽样：记账已在 IsSolid 处完成（record_pair 那颗），这里整段跳过。
					// ============================================================
					if (sNormalMode == 2 && !block_sample && !carve_separate)
					{
						record_pair(x, y, z);
						// 记账已完成；接触点配额用完 ⇒ 这一格到此为止，本窗口剩下的格不再细算
						//（下一颗遍历格的粗判窗口会覆盖其中一半左右，粗判本来就是抽样）
						if (try_record_block())
							return;
					}

					Vec3 center_cand = cand.mOrigin + (x + 0.5f) * cand.mStep[0] + (y + 0.5f) * cand.mStep[1] + (z + 0.5f) * cand.mStep[2];

					// ---- 接触几何：法线方向（见 VoxelShape.h 的 sNormalMode）----
					//
					// sNormalMode == 0：法线/深度用**整网格 SAT 的那根全局轴**。不能逐对吸附到
					//   球心连线或它的主导轴：深穿透交错（掉落卡住）时逐对轴会互相矛盾 —— 有的格对往上推、
					//   有的往下推，求解器净推进力抵消 ⇒ 卡死在穿透位置。全局轴 = 整个网格 OBB 的最小重叠轴，
					//   所有接触方向一致 ⇒ 行为与实心大盒一致（卡死状态时全局轴正好指向"顶出来"的方向）。
					//
					// sNormalMode == 1（对照/诊断）：法线 = **逐对格心连线**（球语义），方向取"从 shape1 那
					//   一格指向 shape2 那一格"，与全局轴同约定（都指向 center2）。此时 |Δc·法线| = |Δc|，
					//   深度退化成两球的中心距公式 r1+r2-|Δc|。半径仍沿用投影半径（那是另一个旋钮）。
					//   ★ 实测：深嵌交错**不卡死、不倾倒**，只是上块沉 0.11 m（≈1.1 格）停住；半格错位
					//     堆叠被斜法线的横向分量横推（低 1.3 cm）。见 `VoxelShapeNormalModeSimulation`。
					//
					// ★ 四种模式下**格对集合完全相同**（跨格枚举 + 球心距筛选都跟法线无关）⇒ 对照是严格的。
					Vec3 contact_axis = axis_world;
					bool use_global4 = false;		// 诊断用：本次接触走的是哪条分支
					bool pure_face4 = false;
					bool m1_global = false;		// mode 1：本对格走了「八格全实心 -> 全局轴」分支（深度/接触点按全局轴算）

					// ★ 2x2x2 邻域八格全实心（整格埋住）⇒ 用整网格全局轴；其余情况保持球语义（格心连线）。
					//   埋住时没有「表面朝哪」可言，局部法线只会互相抵消；且两侧格心已经重叠到一起，
					//   连线方向对数值误差极敏感 ⇒ 法线每帧乱跳。全局轴 = 整网格 OBB 的最短重叠轴，方向唯一、稳定。
					//   浅接触 / 表面格不受影响，法线仍是格心连线。
					//   （已删「深穿透回退 deep4」：那条按「重叠 > 0.25 格」触发，而 Jolt slop 0.02 m = 0.32 格
					//    ⇒ 静止接触全部命中。现在只看「是不是八格全实心」，不看重叠深度。）
					if (sNormalMode == 1)
					{
						if (neighborhood_full8(cand.mVoxel, x, y, z, inCC))
						{
							m1_global = true;
							if (sDiagBranchEnabled)
								sDiagBranch[6].fetch_add(1, std::memory_order_relaxed);	// 与 mode 2 的 n4==8 共用一格计数
						}
						else
						{
							m1_global = false;
							Vec3 to_shape2 = iterate_is_1? (center_cand - inCenterIter) : (inCenterIter - center_cand);
							float length = to_shape2.Length();
							if (length > 1.0e-6f)
								contact_axis = to_shape2 / length;
						}
					}
					else if (sNormalMode == 2)
					{
						// ★ 已删「深重叠短路（deep4）」：它是给 **mode 1** 的逐对格心连线打的补丁
						//   （深嵌时逐对方向互相抵消 ⇒ 推不开）。mode 2 的 2x2x2 邻域分类按邻域
						//   排布定方向，不是格心连线，不存在抵消 ⇒ 短路只会把局部法线整批换成
						//   axis_world，删掉它没有任何收益损失。
						//   ★ 误触发的实测：静止接触穿透 = Jolt slop 0.02 m = **0.32 格**，
						//   而阈值是 0.25 格 ⇒ 球坐在甜甜圈孔口时每一对静止格都命中，径向恢复力
						//   被抹成纯竖直，偏 3 cm 就顺着孔壁滑出去；关掉后同一场景 x 0.03→−0.001
						//   坐稳。见 `VoxelShapeDonutBallSeatDiag`。
						// ---- mode 2：局部 2x2x2 邻域分类 ----
						//
						// b = 遍历侧这一格，它在候选侧格空间里的格心 = inCC（"格 c 占据 [c, c+1]"那一套坐标）。
						// 全部运算都在这个空间里做：格边长恒为 1、格轴恒对齐 ⇒ 不需要投影矩阵、不需要求交，
						// 只有最后搬回世界时才用一次 cand.mStep 的线性组合。
						const Vec3 b_center4(inCC[0], inCC[1], inCC[2]);
						const float bc[3] = { inCC[0], inCC[1], inCC[2] };
						const int ac[3] = { x, y, z };

						// 1) 圈 2x2x2：以**候选侧这一格**（a）为基准，按 b 的格心落在 a 格心的哪一侧张开。
						//    每轴 2 次比较（共 6 次）⇒ 邻域一定包含 a 格自己，且把 b 跨过的那一格也包进来。
						//    ★ 不能改成"取 b 的 AABB 跨过的格再补一格"：碎块贴地时 b 的 AABB 恰好与格对齐
						//      （[g+1, g+2]），那样会圈成 {g+1, g+2}，**漏掉真正接触的实心格 g** ⇒ 邻域全空。
						int lo4[3];
						for (int k = 0; k < 3; ++k)
							lo4[k] = bc[k] >= float(ac[k]) + 0.5f? ac[k] : ac[k] - 1;

						// 2) 数这 8 个位置上**候选侧**的实心格（位图；按字缓存的实现见 3c 的 build_block_mask）
						//    ★ 缓存：块起点只取决于这三个下标 ⇒ 同一个块直接复用上一次的掩码（8 次位查询 -> 3 次比较）
						int mask4;
						if (sBlockMaskCache != 0 && cache_valid
							&& lo4[0] == cache_lo4[0] && lo4[1] == cache_lo4[1] && lo4[2] == cache_lo4[2])
							mask4 = cache_mask4;
						else
						{
							mask4 = build_block_mask(cand.mVoxel, lo4[0], lo4[1], lo4[2]);
							if (sBlockMaskCache != 0)
							{
								cache_valid = true;
								cache_lo4[0] = lo4[0];
								cache_lo4[1] = lo4[1];
								cache_lo4[2] = lo4[2];
								cache_mask4 = mask4;
							}
						}

						// 8 位 popcount（不引入 <bit>）
						int n4 = mask4 - ((mask4 >> 1) & 0x55);
						n4 = (n4 & 0x33) + ((n4 >> 2) & 0x33);
						n4 = (n4 + (n4 >> 4)) & 0x0F;

						// 第 inBit 个位置的格心（候选侧格空间）
						auto cc4 = [&lo4](int inBit)
						{
							return Vec3(float(lo4[0] + (inBit & 1)) + 0.5f,
										float(lo4[1] + ((inBit >> 1) & 1)) + 0.5f,
										float(lo4[2] + ((inBit >> 2) & 1)) + 0.5f);
						};

						float wsum4 = 0.0f;
						Vec3 acc4 = Vec3::sZero();					// "b 应当离开 a" 的局部方向累加

						// 1 格 -> 球语义：离开该格心
						auto add_sphere4 = [&](int inBit)
						{
							Vec3 d = b_center4 - cc4(inBit);
							float length = d.Length();
							if (length > 1.0e-6f)
							{
								acc4 += d * (1.0f / length);	// 除以同一个数原来编译成 3 次 divss；改成 1 次 divss + 3 次 mul
								wsum4 += 1.0f;
							}
						};

						// 2 格同轴 -> 胶囊：离开线段上的最近点
						//
						// ★ 两端点只差**一根轴**（cNbEdgeBits 里成对的那两位）⇒ 线段长度恒为 1、
						//   方向恒为格轴：Dot = ±(bc_ax - p0_ax)、LengthSq = 1，两个垂直分量直接就是
						//   格心差本身。整段于是退化成"1 次乘 + 1 次 clamp + 3 次减"，不再构造 seg。
						//   ★ 运算顺序保持 `q = p0 + dir*t` 再 `b - q`，不要合并成 `(b - p0) - dir*t`：
						//     浮点不满足结合律，换顺序会差 1 ulp，顺着 acc4 放大。
						auto add_capsule4 = [&](int inBitA, int inBitB, int inAxis)
						{
							const Vec3 p0 = cc4(inBitA);
							const float p0a = inAxis == 0? p0.GetX() : (inAxis == 1? p0.GetY() : p0.GetZ());
							// seg = dir * e_axis（mA 是低位端、mB 是高位端，见 cNbEdgeBits）
							const float dir = (inBitB & (1 << inAxis)) != 0? 1.0f : -1.0f;
							const float t = Clamp(dir * (bc[inAxis] - p0a), 0.0f, 1.0f);
							const Vec3 seg = inAxis == 0? Vec3(dir, 0, 0) : (inAxis == 1? Vec3(0, dir, 0) : Vec3(0, 0, dir));
							// d 必须保持 `b - (p0 + seg*t)` 这个**向量**运算顺序：逐分量标量重写会差 1 ulp
							// （实测：改写成 `(b-p0) - dir*t` 或三目拼分量都会让接触法线出现 1 ulp 差异）。
							const Vec3 d = b_center4 - (p0 + seg * t);
							float length = d.Length();
							if (length > 1.0e-6f)
							{
								acc4 += d * (1.0f / length);	// 除以同一个数原来编译成 3 次 divss；改成 1 次 divss + 3 次 mul
								wsum4 += 2.0f;
							}
						};

						// 4 格同面 -> 面法线，指向空格那一侧（偶数 = 该轴正侧，奇数 = 负侧）
						auto add_face4 = [&](int inFace)
						{
							Vec3 axis = inFace < 2? Vec3::sAxisX() : (inFace < 4? Vec3::sAxisY() : Vec3::sAxisZ());
							acc4 += (inFace & 1) == 0? axis : -axis;
							wsum4 += 4.0f;
						};

						// 3) 按排布分流
						if (n4 == 0 || n4 == 8)
						{
							// 0 = 邻域里没有实心格（理论上不该有接触）；8 = 整格埋住（穿透）-> 全局轴
							use_global4 = true;
							// 诊断：n4==8 与 n4==0 分别计数（sDiagBranch[6]/[7]）。
							// ★ 这个数很重要：n4==8 的接触**完全由整网格 SAT 轴决定** ——
							//   地形按 16³ 分块后，小物体被单个子形状"完全包含"（0.5 m 方块 vs 1 m 子形状）
							//   时 SAT 的最短推出轴会退化成水平方向，于是这一整批接触都变成侧向推力
							//   （实测 dump 回放：240 条接触里 108 条走这里，方块在 3° 斜坡上横漂 1.28 m）。
							if (sDiagBranchEnabled)
								sDiagBranch[n4 == 8? 6 : 7].fetch_add(1, std::memory_order_relaxed);
						}
						else if (n4 >= 6)
						{
							// 空位 1~2 个：把 b 推向空位。1 个 = 点；2 个同轴 = 线段最近点；2 个对角 = 拆成两次单推。
							// ★ e0 / e1 / 是否同轴全是掩码补集的纯函数 ⇒ 直接查 sEmptyPlans，
							//   不再每次 while 扫两遍（最多 16 次位测试）。
							const NbEmptyPlan &ep4 = sEmptyPlans.mPlan[mask4];

							if (ep4.mCount == 1)
							{
								Vec3 d = cc4(ep4.mBit[0]) - b_center4;
								float length = d.Length();
								if (length > 1.0e-6f)
								{
									acc4 += d * (1.0f / length);	// 除以同一个数原来编译成 3 次 divss；改成 1 次 divss + 3 次 mul
									wsum4 += float(n4);
								}
							}
							else if (ep4.mCoaxial)
							{
								// 两个空位同一根轴 -> 连成线段，推向最近点。
								// ★ 同 add_capsule4：轴对齐 ⇒ 线段长恒为 1、方向恒为格轴，只有 1 个分量要 clamp。
								//   这里的方向是"空位 - b"（与胶囊相反），运算顺序同样保持 `q = p0 + dir*t` 再 `q - b`。
								const int ax = ep4.mAxis;
								const Vec3 p0 = cc4(ep4.mBit[0]);
								const float p0a = ax == 0? p0.GetX() : (ax == 1? p0.GetY() : p0.GetZ());
								const float dir = (ep4.mBit[1] & (1 << ax)) != 0? 1.0f : -1.0f;
								const float t = Clamp(dir * (bc[ax] - p0a), 0.0f, 1.0f);
								// seg = dir * e_ax（两个空位只差这一位）⇒ 不必再构造 p1 相减
								const Vec3 seg = ax == 0? Vec3(dir, 0, 0) : (ax == 1? Vec3(0, dir, 0) : Vec3(0, 0, dir));
								const Vec3 d = (p0 + seg * t) - b_center4;
								float length = d.Length();
								if (length > 1.0e-6f)
								{
									acc4 += d * (1.0f / length);	// 除以同一个数原来编译成 3 次 divss；改成 1 次 divss + 3 次 mul
									wsum4 += float(n4);
								}
							}
							else
							{
								// 两个对角空位 -> 逐个推向空位（bit 升序，与旧版的 e0 -> e1 扫描顺序一致）
								const float w4 = float(n4) * 0.5f;		// == float(n4) / 2.0f，2 的幂，逐位相同
								for (int k = 0; k < ep4.mCount; ++k)
								{
									Vec3 d = cc4(ep4.mBit[k]) - b_center4;
									float length = d.Length();
									if (length > 1.0e-6f)
									{
										acc4 += d * (1.0f / length);	// 除以同一个数原来编译成 3 次 divss；改成 1 次 divss + 3 次 mul
										wsum4 += w4;
									}
								}
							}
						}
						else
						{
							// 1~5 格：走预计算表（见文件上方 sNbPlans），逐块算方向、按块内格数加权。
							// 表里的块序与"整面 > 同轴棱 > 单格"的贪心顺序逐条一致 ⇒ 浮点累加顺序不变。
							const NbPlan &plan4 = sNbPlans.mPlan[mask4];

							// ★ 单个整面（n4 == 4 = 平面接触，地面/墙面最常见的那一种）：
							//   方向就是这张面的法线 = 该轴的**单位**格轴。mFaceAxis 每个查询只归一化一次
							//   （见 GridSide 的定义），所以这一支 0 次 sqrt、0 次除法、0 次累加 ——
							//   整条 "acc4 累加 -> /wsum4 -> mStep 三点乘加 -> Length -> 再归一化" 都不走。
							//   这也正是"归一化合并后再算"：把 64 个格对各自的一次归一化，摊成整查询 3 次。
							if (plan4.mCount == 1 && plan4.mBlock[0].mType == cNbFace
								&& cand.mFaceAxis[plan4.mBlock[0].mA >> 1].LengthSq() > 0.5f)
							{
								const Vec3 &face_n4 = cand.mFaceAxis[plan4.mBlock[0].mA >> 1];
								contact_axis = (plan4.mBlock[0].mA & 1) == 0? face_n4 : -face_n4;
								if (iterate_is_1)
									contact_axis = -contact_axis;
								pure_face4 = true;
							}

							if (!pure_face4)
								for (int b4 = 0; b4 < plan4.mCount; ++b4)
								{
									const NbBlock &blk4 = plan4.mBlock[b4];
									switch (blk4.mType)
									{
									case cNbFace:
										add_face4(blk4.mA);
										break;

									case cNbEdge:
										add_capsule4(blk4.mA, blk4.mB, blk4.mAxis);
										break;

									default:
										add_sphere4(blk4.mA);
										break;
									}
								}
						}

						// 4) 局部方向 -> 世界，并按"contact_axis 指向 shape2"的约定定符号
						if (!use_global4 && !pure_face4 && wsum4 > 0.0f)
						{
							Vec3 d_local = acc4 / wsum4;
							Vec3 d_world = cand.mStep[0] * d_local.GetX()
										 + cand.mStep[1] * d_local.GetY()
										 + cand.mStep[2] * d_local.GetZ();
							float length = d_world.Length();
							if (length > 1.0e-6f)
							{
								d_world = d_world * (1.0f / length);
								// acc4 是"b（遍历侧）离开 a（候选侧）"的方向；contact_axis 的约定是从 shape1 指向 shape2
								contact_axis = iterate_is_1? -d_world : d_world;
							}
						}
					}

					// 深度。mode 2 用**格与格的重叠量**，其它模式保持"沿法线的球投影"不变（逐位相同）。
					//
					// ★ mode 2 为什么不能用投影：它的 contact_axis 是"这个格子哪一面朝空"，
					//   和两格心连线并不平行（同层横向邻居正交 ⇒ 投影 0 ⇒ depth 整整一格；
					//   斜对角 ⇒ 0.3 格）。实测碎块掉进自己挖出的洞里（零间隙）时每帧几千条这种
					//   假穿透，方块被洞壁来回顶、角度随机跳 1~13°、位置左右抖。
					//   两个体素格的重叠量在格坐标系里就是 `1 格 - max(|dx|,|dy|,|dz|)`（逐轴最小
					//   分离）：相切/斜对角天然给出 0，与法线方向无关 ⇒ 假穿透直接消失。
					//   0 次开方、0 次点乘（dx/dy/dz 在球面筛选时已经算好），比投影版更便宜。
					float depth;
					if (sNormalMode == 2)
					{
						// dmax 已在格对预判处算好（dmax4），dx/dy/dz 中途不变 ⇒ 直接复用，逐位相同。
						// ★★ 单位换算必须用**格长**，不能用 radius_sum（= 沿全局轴的投影半径）：
						//   格重叠量 `1 - dmax4` 是"格"为单位，乘格长才是米。乘投影半径会随轴变斜
						//   而虚高 —— chunk 化地形里 axis_world = 实心盒心2 − (方块盒 ∩ 单个 chunk 盒)
						//   的中心，轴带水平分量 ⇒ 同一帧同一几何，深度 0.0349 变成 0.0576（1.65×）
						//   ⇒ 求解器每帧多推开 0.4 格 ⇒ 方块弹起悬空、再落下、再弹 = 抖动。
						//   两侧格长不同时取小者（保守，不会虚高）。见 `VoxelShapeChunkTerrainJitterDiag`。
						//   ★ 块抽样：half_sum4 = 块半宽 + 候选半格（逐格 = 1.0，逐位一致）。
						depth = min_cell_len * (half_sum4 - dmax4);
						// 重叠 <= 0（相切 / 分离）不产生接触：留一条 0 深度的接触只会让求解器
						// 在洞壁上反复犹豫（每帧正负翻转），这正是"左右抖"的另一半来源。
						// 另有 0.05 格死区：亚毫米重叠里法线方向仍是"格的表面朝向"（离散），推力方向会在
						// 相邻两轴之间来回跳 ⇒ 物体几乎不动，但速度符号每几帧翻一次（看不见的抖）。
						// 0.05 格（vs=0.0625 时 3 mm）远小于求解器 slop 允许的稳态穿透 0.32 格，
						// 不会让静止物体失去接触/摩擦。
						if (depth <= min_cell_len * 0.05f)
							continue;

					}
					else if (sNormalMode == 1)
					{
						if (m1_global)
							// 八格全实心：法线是全局轴、与格心连线不平行 ⇒ 用沿轴的投影重叠，
							// 不能再用球距离（那会按 |Δc| 算出假穿透：轴横向时 |Δc| 小而半径和近整格）
							// ★ 块抽样：radius_sum_sel = radius_sum + 遍历侧一个投影半径（块 = 2 格宽）。
							depth = radius_sum_sel - abs((center_cand - inCenterIter).Dot(contact_axis));
						else
						{
							// ★ 纯球语义（2026-11-05 用户令）：depth = 内切球半径和 − 格心距，与上面
							//   的球筛（dist_limit_sq）同一个半径 ⇒ 球筛通过 ⇔ depth >= 0，逻辑自洽。
							//   旧版用 axis_radius（L1 支撑半径，八面体）在斜连线上系统性虚高：横向错位
							//   dx 的正对格对**真实重叠 = 0**（AABB 只在边上擦碰），支撑半径却给出
							//   0.09~0.22 格假穿透 + 0.1~0.6 的斜法线（Python 对拍实测）⇒ 每对本无接触
							//   的格对都被斜向强行推开；轴对齐时方块相对网格的错位模式整体一致 ⇒ 净横向
							//   力方向恒定 ⇒ 平地上极限环抖动（用户实测：轴对齐抖、带旋转反而不抖）。
							//   内切球下错位格对 |Δc| > 直径、在球筛处就已被毙掉，假接触不存在；
							//   轻微穿透本来就是球语义的常态，交给 Jolt slop（0.32 格）吸收。
							depth = filter_dist_sel - (center_cand - inCenterIter).Length();
						}
					}
					else
						// ★ 块抽样：radius_sum_sel 同上（mode 1 全局轴分支共用）。
						depth = radius_sum_sel - abs((center_cand - inCenterIter).Dot(contact_axis));
					// ★ 已去掉 -max_separation 的推测性放宽：只有真正重叠（depth > 0）才产接触。
					if (depth < 0.0f)
						continue;

					// mode 0 / 1：相交要等深度算完（投影式）才知道 ⇒ 在这里补记；mode 2 已在上面记过。
					// ★ 块抽样：记账已在 IsSolid 处完成，跳过。
					if (sNormalMode != 2 && !block_sample && !carve_separate)
					{
						record_pair(x, y, z);
						// 同上：记账已完成，接触点配额满 ⇒ 这一格到此为止，本窗口剩下的格不再细算
						if (try_record_block())
							return;
					}

					// ★★ 角色还原：接触点必须按 **shape1 / shape2** 分配，不能跟着"遍历/候选"走。
					//   shape1 一侧的点沿 +全局轴 推到格面，shape2 一侧沿 -全局轴 —— 这个方向是
					//   **角色**的属性（旧版永远遍历 shape1 时，point1 = 格心 + 轴 * 半径、
					//   point2 = 格心 - 轴 * 半径，就是这个约定）。
					//
					//   踩过的坑：如果按"遍历侧沿 +、候选侧沿 -"来写，那么当遍历侧是 shape2 时
					//   两个点会**同时镜像到外侧**（shape1 的点跑到远侧、shape2 的点也跑到远侧）。
					//   Jolt 用这两个点算接触分离量（面为空时直接拿它们当单点流形，见
					//   ManifoldBetweenTwoFaces），镜像后分离量变成"隔了 2 格"（+2*(r1+r2)），
					//   接触被判成超出 mMaxSeparationDistance 的推测性接触直接丢掉 ⇒
					//   **方块笔直穿落**（实测：10^3 方块落到体素地板上，遍历侧取 2 时 y 从 0.5
					//   一路掉到 -17.6；取 1 正常停在 0.5）。而"大网格 vs 小方块"时遍历侧自动选中
					//   格数少的一方，正好会撞上 ⇒ 表现为"角和面相交大概率穿透"。
					Vec3 center1 = iterate_is_1? inCenterIter : center_cand;
					float radius_1 = iterate_is_1? iter.mRadius : cand.mRadius;
					Vec3 center2 = iterate_is_1? center_cand : inCenterIter;
					float radius_2 = iterate_is_1? cand.mRadius : iter.mRadius;
					// 接触点沿法线推到**接触面**上，不能用沿全局轴的投影半径：斜轴时投影半径能到
					// 0.87*cell，把点推到格子外面（方块内部 7 cm），Jolt 拿这样的点算力臂和分离量
					// ⇒ 撞击帧过冲。
					//   mode 2：法线是格轴 ⇒ 半格长就是格面中心；
					//   mode 1 非全局分支：纯球语义（见深度处注释）⇒ 接触点在**内切球面**上 = 半格长；
					//   mode 1 的 n8 全局轴分支 + mode 0：法线是格轴 / 全局轴，投影半径（mRadius
					//   就是沿全局轴算的）才是"沿该方向到格面"的正确量，保持不变。
					//   ★ 块抽样：遍历侧的点到**块的接触面** = 遍历侧 push 翻倍（半格→整格 /
					//     投影半径→2 倍）。接触点落在块面上而不是块内的格面上，位置无偏。
					float push_1 = (sNormalMode == 2 || (sNormalMode == 1 && !m1_global))? 0.5f * cell_len_0 : radius_1;
					float push_2 = (sNormalMode == 2 || (sNormalMode == 1 && !m1_global))? 0.5f * cell_len_1 : radius_2;
					if (block_sample)
					{
						if (iterate_is_1) push_1 += push_1; else push_2 += push_2;
					}
					Vec3 point1 = center1 + contact_axis * push_1;
					Vec3 point2 = center2 - contact_axis * push_2;

					// 诊断：分支 × 法线朝上/朝下计数（见 VoxelShape::sDiagBranch）
					if (sDiagBranchEnabled)
					{
						uint32 br = (use_global4 || m1_global)? 0 : (pure_face4? 2 : 4);
						sDiagBranch[br + (contact_axis.GetY() >= 0.0f? 0 : 1)].fetch_add(1, std::memory_order_relaxed);
					}

					SubShapeID id_cand = cand.mVoxel->EncodeSubShapeID(*cand.mIDCreator, uint(x), uint(y), uint(z)).GetID();
					if (!id_iter_done)
					{
						id_iter = iter.mVoxel->EncodeSubShapeID(*iter.mIDCreator, uint(inX), uint(inY), uint(inZ)).GetID();
						id_iter_done = true;
					}
					SubShapeID id1 = iterate_is_1? id_iter : id_cand;
					SubShapeID id2 = iterate_is_1? id_cand : id_iter;

					// 点接触：刻意**不**填 mShape1Face/mShape2Face。Jolt 在面为空时会退回用
					// mContactPointOn1/2 当单点接触（见 ManifoldBetweenTwoFaces），所以接触不会丢。
					CollideShapeResult result(point1, point2, contact_axis, depth, id1, id2, TransformedShape::sGetBodyID(ioCollector.GetContext()));
					ioCollector.AddHit(result);

					++num_contacts;
					// ★ 块抽样：本块的 1 个代表接触已经产出（每块最多一个，见循环开头的门控）
					contact_done = true;
					if (hot)
						sStatContacts.fetch_add(1, std::memory_order_relaxed);	// 调试统计：实际产出的接触点
				}
		if (probe_timed)
			sCallProbeLoopNs += uint64(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - probe_t0).count());
	};

	// ---- 4b) 遍历：把 4 个遍历格的**固定开销**打包进 Vec4 的 4 条车道一起算 ----
	//
	// 这是**唯一**的路径（曾经并存的标量对照路径已按实测结果删除：跨格批量 10³ −4.9% /
	// 20³ −6.7% / 32³ −13.9%，且与标量逐位相同 —— 所以没必要留一条没人用的分支）。
	//
	// 每条车道 = 一个遍历格。四条车道各自算自己的：
	//     格心 -> 候选侧格坐标 c（3 个仿射式）-> 候选下标区间 lo/hi（floor/ceil + 夹取）
	// 这些量对每格都是**同样的算式、同样的长度**，只是输入下标不同 —— 正好是 SIMD 的形状。
	// 候选循环（长度不定、数据相关）仍然按车道逐格标量跑。
	//
	// ★ 这和"把候选盒摊平成一维再向量化"（试过，+17% 慢）不同：那个向量化的是候选循环，而候选
	//   循环平均只有 1.3~1.8 次，摊平表的开销盖过收益；这里向量化的是**每格固定开销**，那才是
	//   瓶颈。
	//
	// 全部广播常量（每次查询只算一次）
	const Vec4 b_org_x = Vec4::sReplicate(cell_origin.GetX());
	const Vec4 b_org_y = Vec4::sReplicate(cell_origin.GetY());
	const Vec4 b_org_z = Vec4::sReplicate(cell_origin.GetZ());
	const Vec4 b_s0x = Vec4::sReplicate(cell_step[0].GetX()), b_s0y = Vec4::sReplicate(cell_step[0].GetY()), b_s0z = Vec4::sReplicate(cell_step[0].GetZ());
	const Vec4 b_s1x = Vec4::sReplicate(cell_step[1].GetX()), b_s1y = Vec4::sReplicate(cell_step[1].GetY()), b_s1z = Vec4::sReplicate(cell_step[1].GetZ());
	const Vec4 b_s2x = Vec4::sReplicate(cell_step[2].GetX()), b_s2y = Vec4::sReplicate(cell_step[2].GetY()), b_s2z = Vec4::sReplicate(cell_step[2].GetZ());
	const Vec4 b_one = Vec4::sReplicate(1.0f);
	const Vec4 b_eps = Vec4::sReplicate(cCellEpsilon);
	const Vec4 b_lm0 = Vec4::sReplicate(float(max(0, smin[0]))), b_lm1 = Vec4::sReplicate(float(max(0, smin[1]))), b_lm2 = Vec4::sReplicate(float(max(0, smin[2])));
	const Vec4 b_hm0 = Vec4::sReplicate(float(min(size[0] - 1, smax[0]))), b_hm1 = Vec4::sReplicate(float(min(size[1] - 1, smax[1]))), b_hm2 = Vec4::sReplicate(float(min(size[2] - 1, smax[2])));

	// 窗口半宽（候选格单位，逐轴）：
	//   逐格 = 0.5（格心 ± 半格 straddle，与旧版逐位一致）；
	//   块抽样 = 遍历侧块（2x2x2）在候选空间里的**精确** AABB 半宽 Σ_i |cell_step[i][k]| ——
	//   块的 8 个格心相对块心的偏移 = Σ_i (±0.5)·cell_step[i]，逐轴分量的最大绝对值就是它。
	//   旋转的网格（碎块斜着砸）下这一项自动变大，窗口跟着放宽，不会漏。
	const float blk_hw[3] =
	{
		block_sample? abs(cell_step[0].GetX()) + abs(cell_step[1].GetX()) + abs(cell_step[2].GetX()) : 0.5f,
		block_sample? abs(cell_step[0].GetY()) + abs(cell_step[1].GetY()) + abs(cell_step[2].GetY()) : 0.5f,
		block_sample? abs(cell_step[0].GetZ()) + abs(cell_step[1].GetZ()) + abs(cell_step[2].GetZ()) : 0.5f,
	};
	const Vec4 b_hx = Vec4::sReplicate(blk_hw[0]);
	const Vec4 b_hy = Vec4::sReplicate(blk_hw[1]);
	const Vec4 b_hz = Vec4::sReplicate(blk_hw[2]);
	// 遍历侧坐标偏移：逐格取格心（+0.5），块抽样取**块心**（+1.0 = 块角格心再 + 半个块）
	const float fx_add = block_sample? 1.0f : 0.5f;

	int buf_x[4], buf_y[4], buf_z[4];
	Vec3 buf_c[4];
	int nbuf = 0;

	// 一次处理 nbuf（1~4）个遍历格。不足 4 个时后面几道用第 0 格填充 —— 只是白算一遍，不产生接触
	// （车道循环只走到 nbuf）。
	auto flush_batch = [&](int inN)
	{
		float fx[4], fy[4], fz[4], wx[4], wy[4], wz[4];
		for (int i = 0; i < 4; ++i)
		{
			int j = i < inN? i : 0;
			fx[i] = float(buf_x[j]) + fx_add;
			fy[i] = float(buf_y[j]) + fx_add;
			fz[i] = float(buf_z[j]) + fx_add;
			wx[i] = buf_c[j].GetX();
			wy[i] = buf_c[j].GetY();
			wz[i] = buf_c[j].GetZ();
		}

		Vec4 vfx(fx[0], fx[1], fx[2], fx[3]);
		Vec4 vfy(fy[0], fy[1], fy[2], fy[3]);
		Vec4 vfz(fz[0], fz[1], fz[2], fz[3]);

		// 候选侧格坐标 c：4 车道一次算完。结合顺序与标量式 (origin + t0) + t1 + t2 保持一致。
		// ★ 块抽样：c = **块心**（遍历侧 +1.0），run_cell 里的逐对几何也全按块心当"遍历格心"。
		Vec4 cx = b_org_x + vfx * b_s0x + vfy * b_s1x + vfz * b_s2x;
		Vec4 cy = b_org_y + vfx * b_s0y + vfy * b_s1y + vfz * b_s2y;
		Vec4 cz = b_org_z + vfx * b_s0z + vfy * b_s1z + vfz * b_s2z;

		// 候选下标区间：4 车道一次算完（floor/ceil + 夹到实心范围）。逐项与标量式一致。
		// ★ 已去掉原来的 ±mExpand 推测余量：区间就是纯粹的"心 ± 半宽"straddle。
		//   逐格半宽 0.5 / 块半宽 Σ|cell_step|（见上面的 blk_hw）。
		Vec4 lo0 = Vec4::sMax(sFloor4(cx - b_hx + b_eps), b_lm0);
		Vec4 hi0 = Vec4::sMin(sCeil4(cx + b_hx - b_eps) - b_one, b_hm0);
		Vec4 lo1 = Vec4::sMax(sFloor4(cy - b_hy + b_eps), b_lm1);
		Vec4 hi1 = Vec4::sMin(sCeil4(cy + b_hy - b_eps) - b_one, b_hm1);
		Vec4 lo2 = Vec4::sMax(sFloor4(cz - b_hz + b_eps), b_lm2);
		Vec4 hi2 = Vec4::sMin(sCeil4(cz + b_hz - b_eps) - b_one, b_hm2);

		Float4 a_lo0, a_hi0, a_lo1, a_hi1, a_lo2, a_hi2, a_cx, a_cy, a_cz;
		lo0.StoreFloat4(&a_lo0); hi0.StoreFloat4(&a_hi0);
		lo1.StoreFloat4(&a_lo1); hi1.StoreFloat4(&a_hi1);
		lo2.StoreFloat4(&a_lo2); hi2.StoreFloat4(&a_hi2);
		cx.StoreFloat4(&a_cx); cy.StoreFloat4(&a_cy); cz.StoreFloat4(&a_cz);

		for (int lane = 0; lane < inN && still_walking(); ++lane)
		{
			int l0 = int(a_lo0[lane]), h0 = int(a_hi0[lane]);
			int l1 = int(a_lo1[lane]), h1 = int(a_hi1[lane]);
			int l2 = int(a_lo2[lane]), h2 = int(a_hi2[lane]);
			if (l0 > h0 || l1 > h1 || l2 > h2)
				continue;								// 该格在候选侧里没有任何候选

			const float cc[3] = { a_cx[lane], a_cy[lane], a_cz[lane] };
			const int lo[3] = { l0, l1, l2 };
			const int hi[3] = { h0, h1, h2 };
			run_cell(buf_x[lane], buf_y[lane], buf_z[lane], buf_c[lane], cc, lo, hi);
		}
	};

	// ================================================================
	// ★★★ 挖洞记账：与「接触点 / 法线」彻底分离的独立 4³ 块通路（2026-11-07）
	//   只跑**一次**（不是每个 2³ 遍历块一次）。推进单位 = 4³ 块 = 位图里正好 1 个 uint64：
	//     · 块内有无实心 → 一次 GetWord，块内 64 格绝不逐格遍历
	//     · 块是否与对方重叠 → 一次块心距（沿用旧 block4 的 lim4 判据）
	//     · 每块写 1 条带 bit31 标记的块号，两侧各一条
	//   之后的所有接触遍历（flush_batch / run_cell / visit_chunk）里零记账，
	//   接触配额与记账配额互不牵扯 ⇒ 两条通路彻底分开。
	// ================================================================

	// ---- 2) 估算冲量：必须在 3) 记账之前，且不依赖记账结果 ----
	//   用 1c) 的全局轴对速度向量做投影，速度 / 逆质量 / 弹性来自步首 body 快照。
	//   < 0 = 这对没有提前判破资格 ⇒ 下面照常记账 + 接触遍历。
	// ★ 整对 body 键（无序 bodyID），用于跨子 shape 累加 Σ承压 / 同帧早退。
	uint64 early_bpk = 0;
	{
		const VoxelStepBody *eb1 = sStepBodyLookup(inVoxel1);
		const VoxelStepBody *eb2 = sStepBodyLookup(inVoxel2);
		if (eb1 != nullptr && eb2 != nullptr)
			early_bpk = sBodyPairKey(eb1->mBodyID, eb2->mBodyID);
	}

	float early_impulse = -1.0f;
	if (sStepBreakEnabled && carve_separate)
		early_impulse = sEstimateEarlyImpulse(inVoxel1, inVoxel2, axis_world);

	if (carve_separate)
	{
		const int *ir = range[iterate_side];
		const VoxelShape *iv = iter.mVoxel;
		const VoxelShape *cv = cand.mVoxel;
		// ★ 球筛半径与接触路径**解耦**：不用 filter_lim（它在小重叠时少一个遍历半格），
		//   一律用块口径 dist_limit + 半块 + 4³ 块半宽 2 + 窗口余量 0.5。宁多记一圈块，
		//   不能漏记 —— 漏一块就是洞缺 4³，正是穿透累积的入口。
		const float lim4 = dist_limit + 0.5f * blk_half_cand + 2.5f;
		const float lim4_sq = lim4 * lim4;
		const int cap = int(sMaxCellsPerSide);

		// 候选侧 4³ 块角的合法夹取范围（网格 ∩ 该侧实心 AABB）
		const int cx0 = max(0, smin[0]), cy0 = max(0, smin[1]), cz0 = max(0, smin[2]);
		const int cx1 = min(size[0] - 1, smax[0]), cy1 = min(size[1] - 1, smax[1]), cz1 = min(size[2] - 1, smax[2]);

		for (int bz = ir[2] & ~3; bz <= ir[5]; bz += 4)
			for (int by = ir[1] & ~3; by <= ir[4]; by += 4)
				for (int bx = ir[0] & ~3; bx <= ir[3]; bx += 4)
				{
					if (bx < 0 || by < 0 || bz < 0
						|| bx + 3 >= iv->mSizeX || by + 3 >= iv->mSizeY || bz + 3 >= iv->mSizeZ)
						continue;
					if (iv->GetWord(bx, by, bz) == 0)
						continue;								// 遍历侧整块空：一次读字

					// 遍历侧 4³ 块心 → 候选侧格坐标（与 flush_batch 同一仿射分解，块心 = 角 + 2）
					const Vec3 c = cell_origin
						+ cell_step[0] * (float(bx) + 2.0f)
						+ cell_step[1] * (float(by) + 2.0f)
						+ cell_step[2] * (float(bz) + 2.0f);
					const float ccx = c.GetX(), ccy = c.GetY(), ccz = c.GetZ();

					// 块心 ± lim4 覆盖的候选侧 4³ 块角范围（向下取整再对齐到 4，宁多勿漏）
					const int qx0 = int(std::floor(ccx - lim4)) & ~3, qx1 = int(std::floor(ccx + lim4)) & ~3;
					const int qy0 = int(std::floor(ccy - lim4)) & ~3, qy1 = int(std::floor(ccy + lim4)) & ~3;
					const int qz0 = int(std::floor(ccz - lim4)) & ~3, qz1 = int(std::floor(ccz + lim4)) & ~3;

					bool iter_written = false;
					for (int qz = qz0; qz <= qz1; qz += 4)
						for (int qy = qy0; qy <= qy1; qy += 4)
							for (int qx = qx0; qx <= qx1; qx += 4)
							{
								if (qx < cx0 || qy < cy0 || qz < cz0
									|| qx > cx1 || qy > cy1 || qz > cz1
									|| qx + 3 >= cv->mSizeX || qy + 3 >= cv->mSizeY || qz + 3 >= cv->mSizeZ)
									continue;
								if (cv->GetWord(qx, qy, qz) == 0)
									continue;					// 候选侧整块空：一次读字
								const float ddx = float(qx) + 2.0f - ccx;
								const float ddy = float(qy) + 2.0f - ccy;
								const float ddz = float(qz) + 2.0f - ccz;
								if (ddx * ddx + ddy * ddy + ddz * ddz > lim4_sq)
									continue;
								if (int(inRecord->mCount1) >= cap || int(inRecord->mCount2) >= cap)
									continue;

								const uint32_t cb = sPackCell(uint(qx), uint(qy), uint(qz)) | 0x80000000u;
								if (sDedupFirstHit(inRecord->mDedupGen, cb))
								{
									const float st = cv->GetContactStrengthAtCell(qx, qy, qz);
									if (iterate_is_1) { inRecord->mStrength2 += st; sRecordCell(inRecord, false, cb); }
									else              { inRecord->mStrength1 += st; sRecordCell(inRecord, true,  cb); }
								}
								if (!iter_written)
								{
									const uint32_t ib = sPackCell(uint(bx), uint(by), uint(bz)) | 0x80000000u;
									if (sDedupFirstHit(inRecord->mDedupGen, ib))
									{
										const float st = iv->GetContactStrengthAtCell(bx, by, bz);
										if (iterate_is_1) { inRecord->mStrength1 += st; sRecordCell(inRecord, true,  ib); }
										else              { inRecord->mStrength2 += st; sRecordCell(inRecord, false, ib); }
									}
									iter_written = true;
								}
							}
				}
	}

	// ★ 把本 leaf pair 的 Σ承压并入「整对 body」累加器（一次锁，不在热循环里加锁）。
	//   这样最后一个叶子对跑到的 tot 已是该对跨所有子 shape 的全量之和。
	if (sStepBreakEnabled && early_bpk != 0
		&& (inRecord->mStrength1 > 0.0f || inRecord->mStrength2 > 0.0f))
	{
		const uint sh = uint(early_bpk & (sStrengthShards - 1));
		lock_guard<mutex> lk(sStrengthMutex[sh]);
		VoxelStepBodyPairAccum &acc = sThisStepStrength[sh][early_bpk];
		acc.sSide1 += inRecord->mStrength1;
		acc.sSide2 += inRecord->mStrength2;
	}

	// ---- 3) 提前判破：记账已完成、接触遍历还没开始 ----
	//   ★ 破帧不产出接触流形 ⇒ 求解器自然不参与，和旧路径 SetIsSensor(true) 的物理结果
	//     一致，却省掉整片逐格法线 / 接触点 / 穿透深度（这部分本来就是白做）。
	//   ★ 冲量用 1c) 算出的全局轴对速度向量做投影，不是直接用重量 × 速度。
	//   ★ 拿不到 body 快照（单测 / 旧 dll / 非 4³ 记账路径）⇒ 整段不启用，行为逐字不变。
	//   ★ 判破结论写进破记录表，C# Flush 开头补成 PairEvent ⇒ 挖洞 / 碎块 / 反向冲量照旧。
	if (sStepBreakEnabled && carve_separate
		&& sTryEarlyBreak(inVoxel1, inVoxel2, early_bpk, early_impulse, axis_world, meet_world))
		return;

	// ---- 4c) chunk 级剪枝：按 8^3 chunk 调用遍历，两种整块跳过 ----
	//   1. 候选侧整个映射区间里没有任何实心 chunk（摘要计数全 0）⇒ 这 512 格的固定开销省；
	//      （几何性剪枝：没有重叠就没有可记的，记账/接触都不受影响。）
	//   2. 遍历 chunk "全内部"（自身 + 6 邻 chunk 全实心）且候选侧重叠 chunk 也全内部
	//      ⇒ 块内全部格对都是"内部格 x 内部格"的埋没对（6 邻居全实心、没有暴露面），
	//      不产生有意义的支撑接触（见 sSkipInteriorPairs 的说明），整块跳过。
	//      ★★ 只在非记账路径跳；记账路径必须把埋没对也记下来（撞击破坏要的是完整重叠格集合，
	//      见下面剪枝处的说明）。
	//   摘要无效（未 EnsureDerived）时不剪，行为与旧版一致。子盒交给 sVisitSolidVoxelsInBox，
	//   它自己还有 chunk/子块两级空跳。
	const auto walk_t0 = std::chrono::steady_clock::now();	// 诊断：遍历行走段起点（含探测，快照里减掉）
	{
		const int *ir = range[iterate_side];
		const VoxelShape *iter_voxel = iter.mVoxel;
		const VoxelShape *cand_voxel = cand.mVoxel;
		const int cmin[3] = { ir[0] >> cChunkShift, ir[1] >> cChunkShift, ir[2] >> cChunkShift };
		const int cmax[3] = { ir[3] >> cChunkShift, ir[4] >> cChunkShift, ir[5] >> cChunkShift };
		// 候选 chunk 区间的每轴余量：straddle 是"格心 ± 半格"，整块跨度再留 1 格余量 ⇒ 2（无推测余量）
		const int pad[3] = { 2, 2, 2 };
		// 候选侧 chunk 维度（夹候选区间用；摘要无效时不算剪枝，只夹进网格即可）
		const int cd_ca[3] =
		{
			(cand_voxel->mSizeX + cChunkMask) >> cChunkShift,
			(cand_voxel->mSizeY + cChunkMask) >> cChunkShift,
			(cand_voxel->mSizeZ + cChunkMask) >> cChunkShift,
		};
		const bool cand_summary = cand_voxel->mDerivedValid && !cand_voxel->mChunkSolid.empty();
		const bool cand_tight = cand_summary && cand_voxel->mChunkTight.size() == cand_voxel->mChunkSolid.size();
		const float vs_it = iter_voxel->mVoxelSize;
		const float hx_it = iter_voxel->mHalfExtent.GetX(), hy_it = iter_voxel->mHalfExtent.GetY(), hz_it = iter_voxel->mHalfExtent.GetZ();

		// inLo3/inHi3 = 这个子盒的格下标界（直接路径传 range，chunk 路径传 chunk ∩ range）。
		auto visit_chunk = [&](const AABox &inSub, const int (&inLo3)[3], const int (&inHi3)[3])
		{
			// ★★★ B 方案块抽样：不逐格枚举，改枚举**块角格**（全局偶数下标）。
			//   块 = [角, 角+1]³ 的 8 格；角格实心直接处理，角格为空时再查其余 7 格
			//   （凹口的块角常常是空的，只要块里还有实心格就得处理 —— 记账在 run_cell 里做，
			//   不依赖角格本身实心）。chunk 是 8 对齐 ⇒ chunk 子盒内的偶数角恰好把子盒铺满、
			//   跨 chunk 不重不漏；被 4c 剪枝跳掉的 chunk 也不会留下半块（块角落在被剪 chunk、
			//   格落在候选 chunk 的块，由候选 chunk 那一次的角枚举处理）。
			if (block_sample)
			{
				const Vec3 &bo = iter.mOrigin;
				const Vec3 &bs0 = iter.mStep[0], &bs1 = iter.mStep[1], &bs2 = iter.mStep[2];
				for (int z = inLo3[2] & ~1; z <= inHi3[2]; z += 2)
					for (int y = inLo3[1] & ~1; y <= inHi3[1]; y += 2)
						for (int x = inLo3[0] & ~1; x <= inHi3[0]; x += 2)
						{
							if (!still_walking())
								return;
							if (!iter_voxel->IsSolid(x, y, z))
							{
								bool any = false;
								for (int k = 1; k < 8 && !any; ++k)
									any = iter_voxel->IsSolid(x + (k & 1), y + ((k >> 1) & 1), z + ((k >> 2) & 1));
								if (!any)
									continue;
							}
							buf_x[nbuf] = x;
							buf_y[nbuf] = y;
							buf_z[nbuf] = z;
							// 块心（世界）：格心表达式（mOrigin + (下标+0.5)*步进）再 + 半个块
							buf_c[nbuf] = bo + float(x + 1) * bs0 + float(y + 1) * bs1 + float(z + 1) * bs2;
							if (++nbuf == 4)
							{
								flush_batch(4);
								nbuf = 0;
								if (!still_walking())
									return;
							}
						}
				return;
			}
			// 表面过滤只在 sInteriorSkipPairs == 2 时启用（与 run_cell 的格对过滤同开关）：
			// mode 0/1 下内部格照常枚举，语义与旧版逐位一致。
			// ★★ 记账路径（inRecord != nullptr）**必须全格枚举**（2026-11-05 用户令）：表面枚举会把
			//   遍历侧的内部格整片跳掉 —— 深穿透时对方那一片格只能靠内部格的窗口够到，跳掉它们
			//   = 中间整片记不到账（削不掉、最后被顶出来）。接触集合不变：多枚举出来的内部格对
			//   会被上面那条"内部格过滤"在格对级跳掉（先记账再 continue），一个接触都不多产。
			auto visit_impl = [&](auto inSurfaceTag)
			{
				iter_voxel->sVisitSolidVoxelsInBox<true, decltype(inSurfaceTag)::value>(inSub, iter.mScale, iter.mTransform,
					[&](int inX, int inY, int inZ, const Mat44 &inCellTransform, Vec3)
					{
						if (!still_walking())
							return;

						buf_x[nbuf] = inX;
						buf_y[nbuf] = inY;
						buf_z[nbuf] = inZ;
						buf_c[nbuf] = inCellTransform.GetTranslation();
						if (++nbuf == 4)
						{
							flush_batch(4);
							nbuf = 0;
							if (!still_walking())
								return;
						}
					});
			};
			if (sInteriorSkipPairs == 2 && inRecord == nullptr)
				visit_impl(std::integral_constant<bool, true>{});
			else
				visit_impl(std::integral_constant<bool, false>{});
		};

		// ★ 遍历范围本身就很小（≤ 2 个 chunk，比如 8^3 碎块）时直接整段遍历：
		//   chunk 剪枝的角点映射 / 摘要扫描在微型查询上是纯开销（实测 8^3 浅接触 +0.6 us），
		//   而小区间里也没有多少可剪的东西。
		const long long total_chunks =
			(long long)(cmax[0] - cmin[0] + 1) * (cmax[1] - cmin[1] + 1) * (cmax[2] - cmin[2] + 1);

		// ---- 4c-pre) 薄板列枚举（遍历侧带旋转，见 2b 的说明）----
		//
		// 每列把世界 x/y/z 三条约束各解成一个一维仿射窗口求交，只访问格心落在 meet 盒内的格子。
		// 格心表达式与 sVisitSolidVoxelsInBox 的 visit_cell 逐位一致（同一基点、同一加法顺序），
		// 候选区间/球面筛选交由同一个 flush_batch / run_cell 处理 ⇒ 每个格对的处理与原路径逐位相同。
		// 提前退出（接触数满）从 lambda return，落回函数末尾公共的 flush_batch。
		if (use_columns)
		{
			[&]()
			{
				const int *rr = range[iterate_side];
				const float wmargin = iter.mVoxel->mVoxelSize;	// diag
				const float wlo[3] = { meet_world.mMin.GetX(), meet_world.mMin.GetY(), meet_world.mMin.GetZ() };
				const float whi[3] = { meet_world.mMax.GetX(), meet_world.mMax.GetY(), meet_world.mMax.GetZ() };

				// 与 sVisitSolidVoxelsInBox 相同的格心基点（同一表达式 ⇒ 同一位型）
				const Vec3 visit_origin = iter.mOrigin + 0.5f * (iter.mStep[0] + iter.mStep[1] + iter.mStep[2]);

				// kax = 与世界 Y 最对齐的列轴（|步进·Y| 最大 ⇒ 每列解出的区间最短）；a0/a1 = 两个列轴
				const float gy[3] = { abs(iter.mStep[0].GetY()), abs(iter.mStep[1].GetY()), abs(iter.mStep[2].GetY()) };
				const int kax = gy[0] > gy[1]? (gy[0] > gy[2]? 0 : 2) : (gy[1] > gy[2]? 1 : 2);
				const int a0 = kax == 0? 1 : 0;
				const int a1 = kax == 2? 1 : 2;
				const float so[3] = { iter.mOrigin.GetX(), iter.mOrigin.GetY(), iter.mOrigin.GetZ() };
				const float sa[3] = { iter.mStep[a0].GetX(), iter.mStep[a0].GetY(), iter.mStep[a0].GetZ() };
				const float sb[3] = { iter.mStep[a1].GetX(), iter.mStep[a1].GetY(), iter.mStep[a1].GetZ() };
				const float sk[3] = { iter.mStep[kax].GetX(), iter.mStep[kax].GetY(), iter.mStep[kax].GetZ() };

				// 一维仿射窗口：v(t) = inBase + (t + 0.5) * inG ∩ [inLo, inHi]，收缩 [ioT0, ioT1]
				auto solve_window = [](float inBase, float inG, float inLo, float inHi, int &ioT0, int &ioT1)
				{
					if (abs(inG) < 1.0e-12f)
					{
						if (inBase < inLo || inBase > inHi)	// 列方向与该轴平行：整列同值，要么全要要么全不要
							ioT1 = ioT0 - 1;
						return;
					}
					float f0 = (inLo - inBase) / inG - 0.5f;
					float f1 = (inHi - inBase) / inG - 0.5f;
					if (f0 > f1) { float s = f0; f0 = f1; f1 = s; }
					f0 = max(f0, float(ioT0) - 1.0f);		// 先夹进现有窗口再取整（防极端值溢出 int）
					f1 = min(f1, float(ioT1) + 1.0f);
					int lo = int(ceil(f0 - 1.0e-4f));		// 微量外扩：边界格心不因舍入丢失
					int hi = int(floor(f1 + 1.0e-4f));
					if (lo > ioT0) ioT0 = lo;
					if (hi < ioT1) ioT1 = hi;
				};

				// meet 盒 8 角点映到遍历侧格下标，取两个列轴分量的 min/max ⇒ 列范围的紧边界
				//（斜薄板沿列轴方向的投影 ≈ 接触脚印，比肥盒的 ij 范围小一个数量级。
				//  凸体顶点投影的包围盒 = 顶点坐标的 min/max，这里是精确的。）
				const Mat44 world_to_cell = Mat44::sScale(iter.mInvScale) * iter.mTransform.Inversed();
				float cmin2[2] = { 1.0e30f, 1.0e30f };
				float cmax2[2] = { -1.0e30f, -1.0e30f };
				for (int corner = 0; corner < 8; ++corner)
				{
					Vec3 p = Vec3(corner & 1? whi[0] : wlo[0], corner & 2? whi[1] : wlo[1], corner & 4? whi[2] : wlo[2]);
					// p = to_world((c + 0.5) * vs - halfExtent) ⇒ c = (to_cell(p) + halfExtent) * invVs - 0.5
					Vec3 c = (world_to_cell * p + iter.mVoxel->mHalfExtent) * iter.mInvVoxelSize - Vec3::sReplicate(0.5f);
					const float cv[2] = { a0 == 0? c.GetX() : a0 == 1? c.GetY() : c.GetZ(),
										  a1 == 0? c.GetX() : a1 == 1? c.GetY() : c.GetZ() };
					for (int m = 0; m < 2; ++m)
					{
						cmin2[m] = min(cmin2[m], cv[m]);
						cmax2[m] = max(cmax2[m], cv[m]);
					}
				}
				const int col_axis[2] = { a0, a1 };
				int col_lo[2], col_hi[2];
				for (int m = 0; m < 2; ++m)
				{
					const int rlo = rr[col_axis[m]], rhi = rr[col_axis[m] + 3];
					cmin2[m] = max(cmin2[m], float(rlo - 2));	// 先夹进格范围再取整（防极端值）
					cmax2[m] = min(cmax2[m], float(rhi + 2));
					col_lo[m] = max(rlo, int(floor(cmin2[m])) - 1);
					col_hi[m] = min(rhi, int(ceil(cmax2[m])) + 1);
				}

				int cell[3];
				auto visit_column = [&](int inI, int inJ)
				{
					// ★ 块抽样：块角格三轴都是全局偶数下标 ⇒ 奇数列里没有角格，整列跳过
					if (block_sample && (((inI & 1) | (inJ & 1)) != 0))
						return;
					if (!still_walking())
						return;

					int t0 = rr[kax], t1 = rr[kax + 3];
					const float fi = float(inI) + 0.5f, fj = float(inJ) + 0.5f;
					if (false) { }	// diag: no window
					else for (int w = 0; w < 3; ++w)
					{
						solve_window(so[w] + fi * sa[w] + fj * sb[w], sk[w], wlo[w] - wmargin, whi[w] + wmargin, t0, t1);
						if (t0 > t1)
							return;
					}
					// ★ 块抽样：块角格 = t 的全局偶数下标；窗口下沿碰到奇数时退一格把角包进来
					//   （那个块的低半在窗口外，但窗口内的格仍要记到账）
					const int t_start = block_sample? (t0 & ~1) : t0;
					for (int t = t_start; t <= t1; ++t)
					{
						if (block_sample && (t & 1))
							continue;
						cell[a0] = inI;
						cell[a1] = inJ;
						cell[kax] = t;
					if (sHotStatsEnabled.load(std::memory_order_relaxed))
					{
						sStatVisitedCells.fetch_add(1, std::memory_order_relaxed);	// 调试统计：遍历过的格（含空格）
						sStatColumnCells.fetch_add(1, std::memory_order_relaxed);	// 诊断：列枚举路径访问的格
					}
					if (block_sample)
					{
						// ★ 块抽样：8 格里任一实心才处理（凹口块角为空的同款兜底）；不跑表面过滤
						//   （记账要全格，run_cell 的格对级内部过滤仍然生效）
						bool any8 = false;
						for (int k = 0; k < 8 && !any8; ++k)
							any8 = iter.mVoxel->IsSolid(cell[0] + (k & 1), cell[1] + ((k >> 1) & 1), cell[2] + ((k >> 2) & 1));
						if (!any8)
							continue;
						// 块心（世界）：格心表达式 + 半个块
						Vec3 center = iter.mOrigin + float(cell[0] + 1) * iter.mStep[0] + float(cell[1] + 1) * iter.mStep[1] + float(cell[2] + 1) * iter.mStep[2];
						buf_x[nbuf] = cell[0];
						buf_y[nbuf] = cell[1];
						buf_z[nbuf] = cell[2];
						buf_c[nbuf] = center;
						if (++nbuf == 4)
						{
							flush_batch(4);
							nbuf = 0;
							if (!still_walking())
								return;
						}
						continue;
					}
					if (!iter.mVoxel->IsSolid(cell[0], cell[1], cell[2]))
						continue;
					// 表面枚举（sInteriorSkipPairs == 2）：与 sVisitSolidVoxelsInBox 的 visit_cell
					// 同一过滤（内部格产不出接触，提前跳掉）。EnsureDerived 已在选边前做过。
					if (sInteriorSkipPairs == 2 && !iter.mVoxel->IsVoxelSurface(cell[0], cell[1], cell[2]))
						continue;
						// 格心：与 sVisitSolidVoxelsInBox 的 visit_cell 同一表达式、同一加法顺序
						Vec3 center = visit_origin + float(cell[0]) * iter.mStep[0] + float(cell[1]) * iter.mStep[1] + float(cell[2]) * iter.mStep[2];
						buf_x[nbuf] = cell[0];
						buf_y[nbuf] = cell[1];
						buf_z[nbuf] = cell[2];
						buf_c[nbuf] = center;
						if (++nbuf == 4)
						{
							flush_batch(4);
							nbuf = 0;
							if (!still_walking())
								return;
						}
					}
				};

				// 列环序：由外往内（接触数被 sMaxContactPoints 截断时保留整片外围，
				// 与 sVisitSolidVoxelsInBox<true> 的截断语义一致）
				const int ci = (col_lo[0] + col_hi[0]) >> 1;
				const int cj = (col_lo[1] + col_hi[1]) >> 1;
				const int max_r = max(max(col_hi[0] - ci, ci - col_lo[0]), max(col_hi[1] - cj, cj - col_lo[1]));
				for (int r = 0; r <= max_r && still_walking(); ++r)
				{
					const int x0 = max(col_lo[0], ci - r), x1 = min(col_hi[0], ci + r);
					const int y0 = max(col_lo[1], cj - r), y1 = min(col_hi[1], cj + r);
					if (x0 > x1 || y0 > y1)
						continue;
					for (int i = x0; i <= x1; ++i)
					{
						visit_column(i, y0);
						if (y1 != y0)
							visit_column(i, y1);
					}
					for (int j = y0 + 1; j <= y1 - 1; ++j)
					{
						visit_column(x0, j);
						if (x1 != x0)
							visit_column(x1, j);
					}
				}
			}();
		}
		else if (total_chunks <= 2)
		{
			// 直接用原浮点盒：按格下标重建会比它宽不到一格，浅接触场景多出一层实心格
			//（实测 8^3 浅接触 +0.6 us 就是这一层）
			const int dlo[3] = { ir[0], ir[1], ir[2] };
			const int dhi[3] = { ir[3], ir[4], ir[5] };
			visit_chunk(iter_box[iterate_side], dlo, dhi);
		}
		else for (int cz = cmin[2]; cz <= cmax[2]; ++cz)
			for (int cy = cmin[1]; cy <= cmax[1]; ++cy)
				for (int cx = cmin[0]; cx <= cmax[0]; ++cx)
				{
					int lo[3] =
					{
						max(ir[0], cx << cChunkShift),
						max(ir[1], cy << cChunkShift),
						max(ir[2], cz << cChunkShift),
					};
					int hi[3] =
					{
						min(ir[3], (cx << cChunkShift) + cChunkMask),
						min(ir[4], (cy << cChunkShift) + cChunkMask),
						min(ir[5], (cz << cChunkShift) + cChunkMask),
					};
					if (cand_summary)
					{
						// chunk 的 8 角点映射到候选侧格坐标（cell_origin / cell_step 就是那条仿射链），
						// 取 min/max + straddle 余量，得到候选 chunk 区间
						float clo[3] = { 1.0e30f, 1.0e30f, 1.0e30f };
						float chi[3] = { -1.0e30f, -1.0e30f, -1.0e30f };
						for (int corner = 0; corner < 8; ++corner)
						{
							float fi = float((corner & 1) != 0? hi[0] : lo[0]) + 0.5f;
							float fj = float((corner & 2) != 0? hi[1] : lo[1]) + 0.5f;
							float fk = float((corner & 4) != 0? hi[2] : lo[2]) + 0.5f;
							Vec3 c = cell_origin + fi * cell_step[0] + fj * cell_step[1] + fk * cell_step[2];
							for (int k = 0; k < 3; ++k)
							{
								float v = k == 0? c.GetX() : k == 1? c.GetY() : c.GetZ();
								clo[k] = min(clo[k], v);
								chi[k] = max(chi[k], v);
							}
						}

						bool any_candidate = false;
						bool all_interior = true;
						int ccm[3], ccM[3];
						ccm[0] = max(0, ((int(floor(clo[0])) - pad[0]) >> cChunkShift));
						ccm[1] = max(0, ((int(floor(clo[1])) - pad[1]) >> cChunkShift));
						ccm[2] = max(0, ((int(floor(clo[2])) - pad[2]) >> cChunkShift));
						ccM[0] = min(cd_ca[0] - 1, ((int(ceil(chi[0])) + pad[0]) >> cChunkShift));
						ccM[1] = min(cd_ca[1] - 1, ((int(ceil(chi[1])) + pad[1]) >> cChunkShift));
						ccM[2] = min(cd_ca[2] - 1, ((int(ceil(chi[2])) + pad[2]) >> cChunkShift));
						for (int sz2 = ccm[2]; sz2 <= ccM[2]; ++sz2)
							for (int sy2 = ccm[1]; sy2 <= ccM[1]; ++sy2)
								for (int sx2 = ccm[0]; sx2 <= ccM[0]; ++sx2)
								{
									const size_t cidx = size_t(sx2) + size_t(sy2) * cd_ca[0] + size_t(sz2) * cd_ca[0] * cd_ca[1];
									if (cand_voxel->mChunkSolid[cidx] != 0)
									{
										// ★ 紧界精修（格级精度）：chunk 里有实心格还不够，实心格必须真的
										//   落在映射格区间（含 pad 余量）内才算"有候选"。坑壁/悬空楼板
										//   这类"chunk 有格但离映射区间还有距离"的块在这里整块省掉。
										//   pad 与 ccm/ccM 的 chunk 级余量同源，只会多不会少 ⇒ 不丢接触。
										bool hit = true;
										if (cand_tight)
										{
											const uint64 t = cand_voxel->mChunkTight[cidx];
											const int tmin[3] = { int(t & 7), int((t >> 6) & 7), int((t >> 12) & 7) };
											const int tmax[3] = { int((t >> 3) & 7), int((t >> 9) & 7), int((t >> 15) & 7) };
											const int cbase[3] = { sx2 << cChunkShift, sy2 << cChunkShift, sz2 << cChunkShift };
											for (int k = 0; k < 3; ++k)
												if (cbase[k] + tmin[k] > int(ceil(chi[k])) + pad[k]
													|| cbase[k] + tmax[k] < int(floor(clo[k])) - pad[k])
												{
													hit = false;
													break;
												}
										}
										if (hit)
										{
											any_candidate = true;
											if (all_interior && !cand_voxel->IsChunkAllInterior(sx2, sy2, sz2))
												all_interior = false;
										}
									}
								}
					// all_interior 只有在"确实有候选 chunk 且它们全内部"时才为真；any_candidate 为假直接跳
					if (!any_candidate)
						continue;
					// ★★ "全内部 x 全内部 ⇒ 整块跳"只对**非记账路径**成立（2026-11-05 用户令）：
					//   埋没对确实产不出有意义的接触，但深穿透时这一剪会把重叠区的中间整片从
					//   **记账**里剪掉（削不掉、被顶出来）。记账路径且两侧未满 ⇒ 照常遍历，
					//   格对级的内部过滤会先记账再跳接触，接触集合不变，只是多走格记账。
					if (iter_voxel->IsChunkAllInterior(cx, cy, cz) && all_interior
						&& (inRecord == nullptr
							|| (inRecord->mCount1 >= uint(VoxelShape::sMaxCellsPerSide)
							 && inRecord->mCount2 >= uint(VoxelShape::sMaxCellsPerSide))))
						continue;
					}
					AABox sub(Vec3(lo[0] * vs_it - hx_it, lo[1] * vs_it - hy_it, lo[2] * vs_it - hz_it),
						Vec3((hi[0] + 1) * vs_it - hx_it, (hi[1] + 1) * vs_it - hy_it, (hi[2] + 1) * vs_it - hz_it));
					visit_chunk(sub, lo, hi);
				}
	}

	if (nbuf > 0)
		flush_batch(nbuf);
	sCallWalkNs += uint64(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - walk_t0).count());
}

void VoxelShape::sCastConvexVsVoxel(const ShapeCast &inShapeCast, const ShapeCastSettings &inShapeCastSettings, const Shape *inShape, Vec3Arg inScale, const ShapeFilter &inShapeFilter, Mat44Arg inCenterOfMassTransform2, const SubShapeIDCreator &inSubShapeIDCreator1, const SubShapeIDCreator &inSubShapeIDCreator2, CastShapeCollector &ioCollector)
{
	// 这个函数注册在 (任意形状, 体素) 上：inShape 一定是体素形状。
	// inShapeCast 已经表达在体素形状的局部空间（含缩放）里，扫掠范围 = 起点包围盒 ∪ 终点包围盒
	// （cast 过程中形状只平移不旋转，所以起点和终点的包围盒就够了）。
	const VoxelShape *voxel = static_cast<const VoxelShape *>(inShape);

	// 质心偏移：扫描范围/格心都要按盒心变换算（COM == 0 时原样返回）
	Mat44 voxel_com2 = voxel->sToGrid(inCenterOfMassTransform2, inScale);

	AABox swept = inShapeCast.mShapeWorldBounds;
	swept.Encapsulate(AABox(inShapeCast.mShapeWorldBounds.mMin + inShapeCast.mDirection, inShapeCast.mShapeWorldBounds.mMax + inShapeCast.mDirection));

	// 再换算到体素形状的未缩放局部空间
	Vec3 inv_scale(
		1.0f / inScale.GetX(),
		1.0f / inScale.GetY(),
		1.0f / inScale.GetZ());
	AABox local_box;
	local_box.Encapsulate(swept.mMin * inv_scale);
	local_box.Encapsulate(swept.mMax * inv_scale);
	voxel->RecordDebugQueryBox(local_box);		// 调试：面板画"查询框"

	const BoxShape *unit_box = sGetUnitBox();
	voxel->sVisitSolidVoxelsInBox(local_box, inScale, voxel_com2,
		[&inShapeCast, &inShapeCastSettings, &inShapeFilter, &inSubShapeIDCreator1, voxel, inScale, &inSubShapeIDCreator2, unit_box, &ioCollector](int inX, int inY, int inZ, const Mat44 &inCellTransform, Vec3 inCellScale)
		{
			// 把该格搬到原点，于是 shape cast 也要反向平移格心。
			//
			// ★ 这里的平移必须作用在"体素形状的坐标系"里，而不是 cast 自己的坐标系里：
			//   cast 可能是带旋转的，如果按 cast 自己的坐标系平移，格子就会落错位置
			//   （斜着投出去的射线会命中不存在的格子，或者漏掉真正的格子）。
			// inShapeCast 表达在"质心局部空间"，格心也要先减掉质心偏移再缩放
			Vec3 cell_center_in_cast_space = (voxel->GetVoxelCenter(inX, inY, inZ) - voxel->GetCenterOfMass()) * inScale;
			ShapeCast cell_cast = inShapeCast.PostTranslated(-cell_center_in_cast_space);

			SubShapeIDCreator cell_id = voxel->EncodeSubShapeID(inSubShapeIDCreator2, uint(inX), uint(inY), uint(inZ));
			CollisionDispatch::sCastShapeVsShapeLocalSpace(cell_cast, inShapeCastSettings, unit_box, inCellScale, inShapeFilter, inCellTransform, inSubShapeIDCreator1, cell_id, ioCollector);
		});
}

//////////////////////////////////////////////////////////////////////////////////////////
// 注册
//////////////////////////////////////////////////////////////////////////////////////////

void VoxelShape::sRegister()
{
	ShapeFunctions &f = ShapeFunctions::sGet(EShapeSubType::User1);
	f.mConstruct = []() -> Shape * { return new VoxelShape; };
	f.mColor = Color::sOrange;

	// 注册一对形状：
	// - (其他形状, 体素) -> sCollideConvexVsVoxel   （遍历体素的格子）
	// - (体素, 其他形状) -> sCollideVoxelVsVoxel    （遍历体素的格子）
	// - cast 同理，反向的那条用通用的 sReversedCastShape
	auto register_pair = [](EShapeSubType inOther) {
		CollisionDispatch::sRegisterCollideShape(inOther, EShapeSubType::User1, sCollideConvexVsVoxel);
		CollisionDispatch::sRegisterCollideShape(EShapeSubType::User1, inOther, sCollideVoxelVsVoxel);
		CollisionDispatch::sRegisterCastShape(inOther, EShapeSubType::User1, sCastConvexVsVoxel);
		CollisionDispatch::sRegisterCastShape(EShapeSubType::User1, inOther, CollisionDispatch::sReversedCastShape);
	};

	// 所有凸形状（Sphere / Box / Capsule / ... / UserConvex1-8）
	for (EShapeSubType s : sConvexSubShapeTypes)
		register_pair(s);

	// Mesh / HeightField / Plane 是**非凸**形状，它们只把自己注册进"凸类型"的组合里，
	// 所以这里必须手工补上。
	// ★ 漏掉的话在 Release 下会静默命中 CollisionDispatch::sInit() 填的 "Unsupported shape pair" 空实现：
	//   不报错、永远 0 命中，非常难查。
	for (EShapeSubType s : { EShapeSubType::Mesh, EShapeSubType::HeightField, EShapeSubType::Plane })
		register_pair(s);

	// 体素 vs 体素：两边都要遍历，用 sCollideVoxelVsVoxel（它会递归成"单格盒子 vs 体素"）
	CollisionDispatch::sRegisterCollideShape(EShapeSubType::User1, EShapeSubType::User1, sCollideVoxelVsVoxel);

	// 体素 cast 体素：sCastConvexVsVoxel 本身不要求"被投的形状是凸的"——它遍历的是**目标**（体素）的格子，
	// 然后把整条 cast 交给通用分发，于是会递归成"（体素）cast 单格盒子" -> "（单格盒子）cast 体素"，仍然正确。
	CollisionDispatch::sRegisterCastShape(EShapeSubType::User1, EShapeSubType::User1, sCastConvexVsVoxel);
}

JPH_NAMESPACE_END
