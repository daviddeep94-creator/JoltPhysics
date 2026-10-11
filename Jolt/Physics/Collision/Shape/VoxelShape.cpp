// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#include <Jolt/Physics/Collision/Shape/VoxelShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/PolyhedronSubmergedVolumeCalculator.h>
#include <Jolt/Physics/Collision/CollisionDispatch.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/CollidePointResult.h>
#include <Jolt/Physics/Collision/CollideSoftBodyVertexIterator.h>
#include <Jolt/Physics/Collision/TransformedShape.h>
#include <Jolt/Core/Profiler.h>
#ifdef JPH_DEBUG_RENDERER
	#include <Jolt/Renderer/DebugRenderer.h>
#endif // JPH_DEBUG_RENDERER

JPH_NAMESPACE_BEGIN

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
/// outSubmergedVolume 是"单位立方体被切出来的体积"（0..1），调用方再乘上格子的真实体积。
/// 返回 false 表示"整格都在水面之上"（什么都不用累加）。
static bool sGetCellSubmergedVolume(Mat44Arg inCellTransform, const Plane &inSurface, float &outSubmergedVolume, Vec3 &outCenterOfBuoyancy JPH_IF_DEBUG_RENDERER(, RVec3Arg inBaseOffset))
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

	// 早退：整格都在水面之下
	if (calculator.AreAllBelow())
	{
		outSubmergedVolume = 1.0f; // 单位立方体的体积是 1
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
};

/// 构造"整个体素网格"的 OBB。
/// 缩放作用在形状自己的局部轴上，所以第 i 根轴上的半长 = 0.5 * size[i] * voxelSize * |scale[i]|；
/// 旋转只改变轴的方向（轴取变换矩阵 3x3 部分的列，平移单独存）。
static VoxelOBB sGetGridOBB(int inSizeX, int inSizeY, int inSizeZ, float inVoxelSize, Vec3Arg inScale, Mat44Arg inCenterOfMassTransform)
{
	VoxelOBB obb;
	obb.mCenter = inCenterOfMassTransform.GetTranslation();
	for (int i = 0; i < 3; ++i)
		obb.mAxis[i] = inCenterOfMassTransform.GetColumn3(i).Normalized();
	obb.mHalf = Vec3(
		0.5f * float(inSizeX) * inVoxelSize * abs(inScale.GetX()),
		0.5f * float(inSizeY) * inVoxelSize * abs(inScale.GetY()),
		0.5f * float(inSizeZ) * inVoxelSize * abs(inScale.GetZ()));
	return obb;
}

/// 整个 OBB 在世界空间里的轴对齐包围盒（8 个角点求包围盒）。
/// 窄相位用它把"对方网格在己方格坐标系里的重叠范围"圈出来。
static AABox sObbWorldBox(const VoxelOBB &inObb)
{
	AABox box;
	for (int i = 0; i < 8; ++i)
	{
		Vec3 corner = inObb.mCenter;
		for (int k = 0; k < 3; ++k)
			corner += inObb.mAxis[k] * (((i >> k) & 1) != 0? inObb.mHalf[k] : -inObb.mHalf[k]);
		box.Encapsulate(corner);
	}
	return box;
}

/// 两个 OBB 的最小分离轴（分离轴定理：候选轴 = 两个盒子各自的 3 根面轴 + 9 根边叉积轴，共 15 根）。
///
/// 返回值是沿这根轴的**重叠量**：正数 = 重叠这么深，0 = 刚好贴合，负数 = 分离了这么多。
/// outAxis 是单位向量，方向统一为"从 A 指向 B"（也就是把 B 推离 A 的方向）。
///
/// ★ 这里刻意**不**在"重叠量为 0 或负数"时提前返回 —— 调用方需要知道"到底分离了多少"，
///   才能把"相距不超过 mMaxSeparationDistance 也算接触"（推测性接触）实现出来。
///   早返回的后果很严重：物体在真正接触之前完全收不到接触，会一路插进去
///   mPenetrationSlop 那么深才被挡住（实测方块会沉 2 厘米，而实心盒不会）。
static float sObbMinSeparationAxis(const VoxelOBB &inA, const VoxelOBB &inB, Vec3 &outAxis, float &outSecondOverlap)
{
	Vec3 delta = inB.mCenter - inA.mCenter;			// 从 A 的盒心指向 B 的盒心

	// 收集候选轴并顺手去重：两个网格朝向一致时，9 根叉积轴全部与面轴重合，
	// 去重后候选数从 15 降到 3，省掉 12 次投影计算。
	Vec3 axes[15];
	int num_axes = 0;
	auto add_axis = [&axes, &num_axes](Vec3Arg inAxis)
	{
		float length = inAxis.Length();
		if (length < 1.0e-6f)						// 两根轴平行时叉积是 0，跳过
			return;
		Vec3 axis = inAxis / length;
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

	float best_overlap = FLT_MAX;
	float second_overlap = FLT_MAX;
	Vec3 best_axis = Vec3::sAxisY();

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
			best_axis = distance >= 0.0f? axis : -axis;
		}
		else if (overlap < second_overlap)
			second_overlap = overlap;
	}

	outAxis = best_axis;
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
/// 逐格接触的数量正比于重叠区的格子数，两个大网格深穿透时会非常多；超过这个数就停止遍历，
/// 免得一次窄相位就吃掉几毫秒。
static constexpr int cMaxVoxelContacts = 1024;

/// 见 VoxelShape.h 里 sUseSphereContacts 的说明。默认开：体素 vs 其它形状每格用内切球。
bool VoxelShape::sUseSphereContacts = true;

/// 见 VoxelShape.h 里 sSkipInteriorPairs 的说明。默认关：内部格 x 内部格的格对照常产出接触。
bool VoxelShape::sSkipInteriorPairs = false;

/// 见 VoxelShape.h 里 sIterateSideOverride 的说明。0 = 自动（遍历重叠区格数少的一侧）。
int VoxelShape::sIterateSideOverride = 0;

/// 体素 vs 体素：接触法线（= 深度方向、接触点推出方向）取自哪里。详见 VoxelShape.h 里的说明书。
///
///   0 = 整网格 SAT 全局轴（历史行为）：所有格对共用一根轴，深嵌时接触方向一致不互相抵消；
///       但曲面/斜面接触给出与接触面无关的方向（甜甜圈落圆锥被顶飞、球放环心自转）。
///   1 = 逐对格心连线（球语义）：n = normalize(格心2 - 格心1)，复用深度那行本来就要算的 Δc，
///       边际成本只有 1 次 Length + 1 次除；但深嵌交错时上块**沉 0.11 m ≈ 1.1 格**（不卡死、
///       不倾倒，只是静止高度错），半格错位堆叠被 45° 斜法线横推（低 1.3 cm）。
///   2 = 暴露面法线：取 shape1 那一格"朝向 shape2 且邻居为空"的格面，法线落在格子局部轴上，
///       每轴只测 sign(d) 那一张面 ⇒ <= 3 次 IsSolid + 3 次 dot；整格埋住（无外露面）退回全局轴
///       ⇒ 继承 mode 0 的深嵌稳定性。实测六个场景方向全对，曲面接触是阶梯法线。
///   3 = 融合模式（"融化"，当前默认）：数 shape1 那一格的外露面（常量方向短路链，数够 2 就停）
///       并顺手记 bitmask —— 外露 >= 2（棱格 / 角格 = 物体边缘）走 mode 1 的格心连线（方向可为
///       任意角度，边缘圆润），外露 < 2（平面格 + 整格埋住）走 mode 2（平面给精确格面法线、
///       整格埋住退回全局轴）。mask 让 mode 2 分支**不再查邻居** ⇒ 平面格正好 6 次查询 + 3 dot。
///       ★ 已知弱点：判据作用在**遍历侧**的格子上；遍历侧本身是一格厚的板时（每格天生上下都
///         外露 = 2）整块退化成 mode 1（实测沉 1.35 cm，见 VoxelShapeNormalModeSimulation）。
///
/// ★ 四个模式**格对集合完全相同**（跨格枚举 + 球心距筛选与法线无关，实测接触数 210/210/210/210、
///   1024 x4）⇒ 是同一次遍历下的严格对照，差异只来自深度筛选那一步。
/// ★ 性能（count-only，四模式同进程交替 7 轮取中位数，`VoxelShapeNormalModeCost`，值 = 相对 mode 2）：
///       网格（接触数）   mode 0   mode 1   mode 2   mode 3
///       10^3（210）      0.89     0.92     1.00     1.02
///       20^3（1024）     0.86     0.88     1.00     0.98
///   ⇒ mode 1 最便宜（复用已算好的 Δc）；**mode 3 与 mode 2 基本持平（20^3 还略快）** —— 见下面
///     那条"分类按格算一次"的优化：融合的路由判据只与那一格自身有关，原来写在格对循环里被
///     重复了（N 个候选 ⇒ N 次 6 邻居查询），提到格级后可摊薄，且 mode 2 分支靠 mask 免查邻居。
///   ★ 别把分类写回格对循环里；也别把"数外露面"塞进 mode 2 的循环体（每轴两张面都探）：实测慢 15%。
int VoxelShape::sNormalMode = 3;

/// 见 VoxelShape.h 里 sLastGridOverlap 的说明。只用于诊断，不参与碰撞逻辑。
float VoxelShape::sLastGridOverlap[2] = { 0.0f, 0.0f };

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
	mVoxels(inSettings.mVoxels),
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

	// 体素数据（形状不拥有它，但必须保证在形状存活期间一直有效）
	if (mVoxels == nullptr)
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

VoxelShape::VoxelShape(int inSizeX, int inSizeY, int inSizeZ, float inVoxelSize, const uint8 *inVoxels, float inDensity) :
	Shape(EShapeType::User1, EShapeSubType::User1),
	mSizeX(inSizeX),
	mSizeY(inSizeY),
	mSizeZ(inSizeZ),
	mVoxelSize(inVoxelSize),
	mVoxels(inVoxels),
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
	if (mNumSolidVoxels == 0)
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
	const int solid_min[3] = { mSolidMin[0], mSolidMin[1], mSolidMin[2] };
	const int solid_max[3] = { mSolidMax[0], mSolidMax[1], mSolidMax[2] };

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

template <bool tOutsideIn, class F>
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

				const uint64 *words = GetChunkWords(chunk_x, chunk_y, chunk_z);
				if (words == nullptr)
					continue;

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

							// 这个 4x4x4 子块里一格实心都没有 -> 一次跳过 64 次 IsSolid
							if (x0 > x1 || y0 > y1 || z0 > z1 || words[sGetSubChunkIndex(x_base, y_base, z_base)] == 0)
								continue;

							for (int iz = 0; iz <= z1 - z0; ++iz)
							{
								int z = index(z0, z1, iz);
								for (int iy = 0; iy <= y1 - y0; ++iy)
								{
									int y = index(y0, y1, iy);

									// ★ 逐格直接按下标读，不再走 IsSolid：
									//   x0..x1 / y0..y1 / z0..z1 是 clamp 进网格后的区间（sGetCellRange 保证），
									//   所以 6 次越界比较 + 每格一次下标乘法都可以省掉，行首下标算一次即可。
									//   读的仍是**实时**的体素字节（不是占用掩码），就地编辑后不需要 Recalculate
									//   也会立刻生效 —— 与 IsSolid 的语义一致。
									const uint8 *row = mVoxels + GetVoxelIndex(x0, y, z);
									for (int ix = 0; ix <= x1 - x0; ++ix)
									{
										int x = index(x0, x1, ix);
										if (row[x - x0] != 0)
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
					// 这一行在掩码字里的位偏移（行内不变）：移位后低 4 位就是该行 4 个 x 位置的实心位
					int shift = ((z & cSubChunkMask) << 4) | ((y & cSubChunkMask) << cSubChunkShift);
					int sub_yz = (((y >> cSubChunkShift) & (cNumSubChunksPerAxis - 1)) << 1)
							   | (((z >> cSubChunkShift) & (cNumSubChunksPerAxis - 1)) << 2);

					const uint8 *row = mVoxels + GetVoxelIndex(min_x, y, z);

					int run_start = -1, run_end = -1;
					uint8 run_value = 0;

					// cx / sx 两层循环让 x 单调递增，游程可以跨 chunk 与子块延续
					for (int cx = min_chunk_x; cx <= max_chunk_x; ++cx)
					{
						int cx_base = cx << cChunkShift;
						int xa = max(min_x, cx_base), xb = min(max_x, cx_base + cChunkSize - 1);

						const uint64 *words = GetChunkWords(cx, cy, cz);
						if (words == nullptr)
						{
							// 整块为空（或越界）：游程在这里断开
							if (run_start >= 0)
							{
								emit_run(run_start, run_end, y, z);
								run_start = -1;
							}
							continue;
						}

						for (int sx = 0; sx < cNumSubChunksPerAxis; ++sx)
						{
							int sub_base = cx_base + (sx << cSubChunkShift);
							int x0 = max(xa, sub_base), x1 = min(xb, sub_base + cSubChunkSize - 1);
							if (x0 > x1)
								continue;												// 该子块不在遍历范围内

							// 这个 4 格组全空 -> 收尾当前游程，并一次跳过 4 格
							if (((words[sx | sub_yz] >> shift) & 0xF) == 0)
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
								// 读的是**实时**体素字节（不是占用掩码），就地编辑后不需要 Recalculate 也立刻生效
								uint8 value = row[x - min_x];
								if (value == 0)
								{
									if (run_start >= 0)
									{
										emit_run(run_start, run_end, y, z);
										run_start = -1;
									}
								}
								else if (run_start >= 0 && value == run_value)
								{
									run_end = x;											// 与当前游程同值 -> 延长
								}
								else
								{
									if (run_start >= 0)
										emit_run(run_start, run_end, y, z);					// 换值（或刚从空开始）-> 收尾旧的，起新的
									run_start = run_end = x;
									run_value = value;
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

void VoxelShape::Recalculate() const
{
	// 半长（未缩放局部空间）
	mHalfExtent = Vec3(0.5f * mSizeX * mVoxelSize, 0.5f * mSizeY * mVoxelSize, 0.5f * mSizeZ * mVoxelSize);

	// 把一个体素坐标塞进 SubShapeID 需要多少位
	mBitsX = sGetBitsForSize(mSizeX);
	mBitsY = sGetBitsForSize(mSizeY);
	mBitsZ = sGetBitsForSize(mSizeZ);

	// 扫一遍体素数据，统计实心格数量 / 表面格数量 / 实心格包围范围
	uint num_solid = 0;
	uint num_surface = 0;
	int min_x = mSizeX, min_y = mSizeY, min_z = mSizeZ;
	int max_x = -1, max_y = -1, max_z = -1;

	if (mSizeX > 0 && mSizeY > 0 && mSizeZ > 0 && mVoxels != nullptr)
		for (int z = 0; z < mSizeZ; ++z)
			for (int y = 0; y < mSizeY; ++y)
				for (int x = 0; x < mSizeX; ++x)
					if (mVoxels[GetVoxelIndex(x, y, z)] != 0)
					{
						++num_solid;

						if (x < min_x) min_x = x;
						if (y < min_y) min_y = y;
						if (z < min_z) min_z = z;
						if (x > max_x) max_x = x;
						if (y > max_y) max_y = y;
						if (z > max_z) max_z = z;

						// 只要有任意一个邻居是空的，这一格就落在表面上。
						// 这个计数是给"只遍历表面格"的优化当上界估计用的，现在只做统计。
						if (IsVoxelFaceExposed(x, y, z, 1, 0, 0) || IsVoxelFaceExposed(x, y, z, -1, 0, 0)
							|| IsVoxelFaceExposed(x, y, z, 0, 1, 0) || IsVoxelFaceExposed(x, y, z, 0, -1, 0)
							|| IsVoxelFaceExposed(x, y, z, 0, 0, 1) || IsVoxelFaceExposed(x, y, z, 0, 0, -1))
							++num_surface;
					}

	mNumSolidVoxels = num_solid;
	mNumSurfaceVoxels = num_surface;
	mSolidMin[0] = min_x; mSolidMin[1] = min_y; mSolidMin[2] = min_z;
	mSolidMax[0] = max_x; mSolidMax[1] = max_y; mSolidMax[2] = max_z;

	// ---- 稀疏分块占用掩码（一个 8x8x8 chunk = 512 bit = 8 个 uint64，一个字一个 4x4x4 子块）----
	//
	// 遍历重叠区时先看掩码：整个 chunk 为空就一次跳过 512 个格子，chunk 不空但某个 4x4x4 子块为空
	// 再跳过 64 格。掩码只占 1 bit / 体素，所以它不是体素数据的副本（那会让内存直接翻倍），
	// 只是给体素图加的一层索引（和 Teardown 的 3D bitmap 一个意思）。
	mNumChunks[0] = (mSizeX + cChunkMask) >> cChunkShift;
	mNumChunks[1] = (mSizeY + cChunkMask) >> cChunkShift;
	mNumChunks[2] = (mSizeZ + cChunkMask) >> cChunkShift;
	mChunkMasks.clear();
	if (num_solid > 0)
	{
		int num_chunks = mNumChunks[0] * mNumChunks[1] * mNumChunks[2];
		mChunkMasks.resize(size_t(num_chunks) * cChunkWords, uint64(0));

		// 只扫实心格的包围范围，不用走遍整张网格
		for (int z = min_z; z <= max_z; ++z)
			for (int y = min_y; y <= max_y; ++y)
				for (int x = min_x; x <= max_x; ++x)
					if (mVoxels[GetVoxelIndex(x, y, z)] != 0)
					{
						int chunk_x = x >> cChunkShift;
						int chunk_y = y >> cChunkShift;
						int chunk_z = z >> cChunkShift;
						size_t chunk_index = size_t(chunk_x + chunk_y * mNumChunks[0] + chunk_z * mNumChunks[0] * mNumChunks[1]);
						mChunkMasks[chunk_index * cChunkWords + sGetSubChunkIndex(x, y, z)] |= uint64(1) << sGetSubChunkBit(x, y, z);
					}
	}

	// ---- 质量属性 ----
	// 每一格是一个均匀的小立方体：质量 = 体积 * 密度，绕自身质心的转动惯量 = m * s^2 / 6（三个轴一样）。
	// 用平行轴定理把它搬到网格中心；因为格子都是轴对齐的，最终惯量张量是对角的。
	float cell_volume = Cubed(mVoxelSize);
	float cell_mass = cell_volume * mDensity;
	float cell_inertia = cell_mass * Square(mVoxelSize) / 6.0f;

	mMassProperties = MassProperties();
	mMassProperties.mMass = float(num_solid) * cell_mass;
	mMassProperties.mInertia = Mat44::sIdentity(); // 没有实心格时给一个单位惯量，避免刚体属性变成 NaN
	if (num_solid > 0 && cell_mass > 0.0f)
	{
		// 只遍历实心格包围范围，不用扫整张网格
		Vec3 inertia_diag = Vec3::sZero();
		for (int z = min_z; z <= max_z; ++z)
			for (int y = min_y; y <= max_y; ++y)
				for (int x = min_x; x <= max_x; ++x)
					if (IsSolid(x, y, z))
					{
						Vec3 r = GetVoxelCenter(x, y, z);
						float r_sq = r.LengthSq();
						inertia_diag += Vec3(
							cell_inertia + cell_mass * (r_sq - r.GetX() * r.GetX()),
							cell_inertia + cell_mass * (r_sq - r.GetY() * r.GetY()),
							cell_inertia + cell_mass * (r_sq - r.GetZ() * r.GetZ()));
					}

		mMassProperties.mInertia = Mat44::sZero();
		mMassProperties.mInertia.SetDiagonal3(inertia_diag);
		mMassProperties.mInertia.SetColumn4(3, Vec4(0, 0, 0, 1));
	}

	// 让缓存的调试几何失效（几何里已经把 mVoxelSize 烘进去了，体素数据一改就得重建）
	++mVersion;
#ifdef JPH_DEBUG_RENDERER
	mGeometry = nullptr;
	mGeometryVersion = uint(-1);
#endif // JPH_DEBUG_RENDERER
}

//////////////////////////////////////////////////////////////////////////////////////////
// 查询接口
//////////////////////////////////////////////////////////////////////////////////////////

const PhysicsMaterial *VoxelShape::GetMaterial(const SubShapeID &inSubShapeID) const
{
	// 整块体素形状共用一个材质。
	// 将来要做"按格换材质"，就在这里先 DecodeSubShapeID 取出坐标，再用 mVoxels[...] 去查材质表。
	return mMaterial.GetPtr();
}

Vec3 VoxelShape::GetSurfaceNormal(const SubShapeID &inSubShapeID, Vec3Arg inLocalSurfacePosition) const
{
	// SubShapeID 里编码了体素坐标，解出来就知道是哪一个格子
	uint x, y, z;
	DecodeSubShapeID(inSubShapeID, x, y, z);

	if (x < uint(mSizeX) && y < uint(mSizeY) && z < uint(mSizeZ) && IsSolid(int(x), int(y), int(z)))
	{
		Vec3 normal;
		if (FindClosestVoxelFace(int(x), int(y), int(z), inLocalSurfacePosition, normal))
			return normal;
	}

	// 解码失败，或者该格被实心邻居完全包围（点在实体深处，没有可以"推出去"的面）：退化成朝上
	return Vec3::sAxisY();
}

void VoxelShape::GetSubmergedVolume(Mat44Arg inCenterOfMassTransform, Vec3Arg inScale, const Plane &inSurface, float &outTotalVolume, float &outSubmergedVolume, Vec3 &outCenterOfBuoyancy JPH_IF_DEBUG_RENDERER(, RVec3Arg inBaseOffset)) const
{
	JPH_PROFILE_FUNCTION();

	// 总排水体积 = 实心格数 * 单格体积
	outTotalVolume = GetVolume();
	outSubmergedVolume = 0.0f;
	outCenterOfBuoyancy = Vec3::sZero();

	if (mNumSolidVoxels == 0 || mVoxelSize <= 0.0f)
		return;

	// 体素形状没有"派生几何"可以整块交给 PolyhedronSubmergedVolumeCalculator，所以逐格算：
	// 每一格都是同一个小立方体，只是位置不同。
	float cell_volume = Cubed(mVoxelSize) * inScale.Abs().GetX() * inScale.Abs().GetY() * inScale.Abs().GetZ();

	// 逐格求浸没体积与浮心（按体积加权平均）
	float total_submerged = 0.0f;
	Vec3 total_moment = Vec3::sZero();
	Mat44 voxel_to_target = inCenterOfMassTransform * Mat44::sScale(inScale);
	for (int z = 0; z < mSizeZ; ++z)
		for (int y = 0; y < mSizeY; ++y)
			for (int x = 0; x < mSizeX; ++x)
				if (IsSolid(x, y, z))
				{
					Mat44 cell_transform = voxel_to_target * Mat44::sTranslation(GetVoxelCenter(x, y, z));

					float cell_submerged_volume; // 0..1，单位立方体被水面切出来的体积
					Vec3 cell_center_of_buoyancy;
					if (sGetCellSubmergedVolume(cell_transform, inSurface, cell_submerged_volume, cell_center_of_buoyancy JPH_IF_DEBUG_RENDERER(, inBaseOffset)))
					{
						float volume = cell_submerged_volume * cell_volume;
						total_submerged += volume;
						total_moment += volume * cell_center_of_buoyancy;
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

	if (mVoxels == nullptr || mNumSolidVoxels == 0)
		return;

	// 限制一次提交的四边形数量，防止网格很大时撑爆内存
	uint num_quads = 0;
	uint max_quads = min(mNumSurfaceVoxels * 6, cMaxDebugQuads);
	outTriangles.reserve(2 * max_quads);

	float half = 0.5f * mVoxelSize;
	for (int z = 0; z < mSizeZ; ++z)
		for (int y = 0; y < mSizeY; ++y)
			for (int x = 0; x < mSizeX; ++x)
				if (IsSolid(x, y, z))
				{
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

	inRenderer->DrawGeometry(inCenterOfMassTransform * Mat44::sScale(inScale), inColor, mGeometry, DebugRenderer::ECullMode::CullBackFace, DebugRenderer::ECastShadow::On, inDrawWireframe? DebugRenderer::EDrawMode::Wireframe : DebugRenderer::EDrawMode::Solid);
}
#endif // JPH_DEBUG_RENDERER

bool VoxelShape::CastRay(const RayCast &inRay, const SubShapeIDCreator &inSubShapeIDCreator, RayCastResult &ioHit) const
{
	JPH_PROFILE_FUNCTION();

	if (mNumSolidVoxels == 0 || mVoxelSize <= 0.0f)
		return false;

	// 三维 DDA（Amanatides & Woo）：沿格子边界一步一步往前走，只访问射线真正穿过的那几格。
	// 射线已经表达在体素形状的**未缩放局部空间**里，格子 c 在该空间里占据
	// [c * s - H, (c + 1) * s - H]。
	const Vec3 &origin = inRay.mOrigin;
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

	if (mNumSolidVoxels == 0 || mVoxelSize <= 0.0f)
		return;

	// 把顶点变到体素形状的未缩放局部空间，找出它落在哪一格；
	// 如果那一格是实心的，就沿"离该点最近的格子面"把顶点推出去。
	Mat44 target_to_voxel = inCenterOfMassTransform.Inversed();
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
			Vec3 contact = inCenterOfMassTransform * (local_contact * inScale);
			Vec3 normal = inCenterOfMassTransform.Multiply3x3(local_normal * inv_scale).Normalized();

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
	// 这个函数注册在 (任意形状, 体素) 上：shape2 一定是体素形状
	const VoxelShape *voxel = static_cast<const VoxelShape *>(inShape2);

	// 用 shape1 的包围盒圈出要遍历的格子范围（变换到体素的未缩放局部空间）。
	//
	// ★ 必须先按 mMaxSeparationDistance 外扩：这个设置的语义是"相距不超过 d 也算接触"。
	//   如果不外扩，就会出现"两个形状的包围盒并不重叠、但距离在 d 以内"的情况 —— 遍历集合是空的，
	//   整片接触被静默跳过（测试里的表现是：把物体抬高 0.03 m 再配 mMaxSeparationDistance = 0.1 就查不到任何接触）。
	//   这与 CollideConvexVsTriangles 的做法一致（它也是先 ExpandBy(mMaxSeparationDistance) 再换算到对方空间）。
	AABox shape1_world_bounds = inShape1->GetWorldSpaceBounds(inCenterOfMassTransform1, inScale1);
	shape1_world_bounds.ExpandBy(Vec3::sReplicate(inCollideShapeSettings.mMaxSeparationDistance));
	AABox local_box = sWorldBoxToVoxelLocal(shape1_world_bounds, inCenterOfMassTransform2, inScale2);

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
		voxel->sVisitSolidVoxelRunsInBox(local_box, inScale2, inCenterOfMassTransform2,
			[inShape1, inScale1, inCenterOfMassTransform1, &inSubShapeIDCreator1, voxel, &inSubShapeIDCreator2, &inCollideShapeSettings, &ioCollector, &inShapeFilter, unit_sphere, sphere_scale](int inX, int inRunLength, int inY, int inZ, const Mat44 &inFirstCellTransform, Vec3Arg inRunCenter, Vec3Arg inRunAxis)
			{
				// 首格的 SubShapeID：游程内体素值相同，取材质/法线用首格都是对的
				SubShapeIDCreator cell_id = voxel->EncodeSubShapeID(inSubShapeIDCreator2, uint(inX), uint(inY), uint(inZ));

				if (inRunLength == 1)
				{
					// 单格游程：还是内切球（胶囊的圆柱半高必须 > 0，单格没有"段"）
					CollisionDispatch::sCollideShapeVsShape(inShape1, unit_sphere, inScale1, sphere_scale, inCenterOfMassTransform1, inFirstCellTransform, inSubShapeIDCreator1, cell_id, inCollideShapeSettings, ioCollector, inShapeFilter);
				}
				else
				{
					// 一根胶囊覆盖整段：局部 Y 轴对齐到游程方向，中心落在游程中点
					const CapsuleShape *capsule = sGetUnitCapsule(inRunLength);
					Mat44 capsule_transform = Mat44::sRotationTranslation(Quat::sFromTo(Vec3::sAxisY(), inRunAxis), inRunCenter);
					CollisionDispatch::sCollideShapeVsShape(inShape1, capsule, inScale1, sphere_scale, inCenterOfMassTransform1, capsule_transform, inSubShapeIDCreator1, cell_id, inCollideShapeSettings, ioCollector, inShapeFilter);
				}
			});
		return;
	}

	voxel->sVisitSolidVoxelsInBox(local_box, inScale2, inCenterOfMassTransform2,
		[inShape1, inScale1, inCenterOfMassTransform1, &inSubShapeIDCreator1, voxel, &inSubShapeIDCreator2, &inCollideShapeSettings, &ioCollector, &inShapeFilter, unit_box, unit_sphere](int inX, int inY, int inZ, const Mat44 &inCellTransform, Vec3 inCellScale)
		{
			// 该格的 SubShapeID：低若干位是 X，然后是 Y，最后是 Z。破坏系统据此知道该删哪一格。
			SubShapeIDCreator cell_id = voxel->EncodeSubShapeID(inSubShapeIDCreator2, uint(inX), uint(inY), uint(inZ));

			if (unit_sphere != nullptr)
			{
				// 内切球路径（非均匀缩放时的兜底）：半径 = 0.5 * min(缩放分量)（与体素 vs 体素格对筛选
				// 阈值里的"半格"同一口径，保证球不超出格子）。球只支持均匀缩放，所以给 sReplicate 的统一缩放。
				// 产出 1 个点接触（面为空时 ManifoldBetweenTwoFaces 退回单点，接触不会丢）。
				Vec3 sphere_scale = Vec3::sReplicate(inCellScale.Abs().ReduceMin());
				CollisionDispatch::sCollideShapeVsShape(inShape1, unit_sphere, inScale1, sphere_scale, inCenterOfMassTransform1, inCellTransform, inSubShapeIDCreator1, cell_id, inCollideShapeSettings, ioCollector, inShapeFilter);
			}
			else
			{
				// 每格就是一个独立的"凸盒子 vs 凸盒子"查询，交给常规凸-凸通道处理
				CollisionDispatch::sCollideShapeVsShape(inShape1, unit_box, inScale1, inCellScale, inCenterOfMassTransform1, inCellTransform, inSubShapeIDCreator1, cell_id, inCollideShapeSettings, ioCollector, inShapeFilter);
			}
		});
}

void VoxelShape::sCollideVoxelVsVoxel(const Shape *inShape1, const Shape *inShape2, Vec3Arg inScale1, Vec3Arg inScale2, Mat44Arg inCenterOfMassTransform1, Mat44Arg inCenterOfMassTransform2, const SubShapeIDCreator &inSubShapeIDCreator1, const SubShapeIDCreator &inSubShapeIDCreator2, const CollideShapeSettings &inCollideShapeSettings, CollideShapeCollector &ioCollector, const ShapeFilter &inShapeFilter)
{
	// 这个函数注册在 (体素, 任意形状) 上：shape1 一定是体素形状。
	const VoxelShape *voxel = static_cast<const VoxelShape *>(inShape1);

	// ---- 情况 A：对方也是体素网格 -> 走手写快路径 ----
	//
	// 这是可破坏世界里最常见、也最吃性能的组合（碎块互相撞）。
	// 手写路径不查碰撞分发表、不为任何格子创建形状对象、也不做 GJK/EPA，而且能给出整片接触共用的法线 ——
	// 后者是"两个方块叠在一起能互相推开"的前提（详细原因见 sCollideVoxelGrids）。
	if (inShape2->GetSubType() == EShapeSubType::User1)
	{
		sCollideVoxelGrids(voxel, inScale1, inCenterOfMassTransform1, inSubShapeIDCreator1,
			static_cast<const VoxelShape *>(inShape2), inScale2, inCenterOfMassTransform2, inSubShapeIDCreator2,
			inCollideShapeSettings, ioCollector);
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
	AABox local_box = sWorldBoxToVoxelLocal(shape2_world_bounds, inCenterOfMassTransform1, inScale1);

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
		voxel->sVisitSolidVoxelRunsInBox(local_box, inScale1, inCenterOfMassTransform1,
			[inShape2, inScale2, inCenterOfMassTransform2, voxel, &inSubShapeIDCreator1, &inSubShapeIDCreator2, &inCollideShapeSettings, &ioCollector, &inShapeFilter, unit_sphere, sphere_scale](int inX, int inRunLength, int inY, int inZ, const Mat44 &inFirstCellTransform, Vec3Arg inRunCenter, Vec3Arg inRunAxis)
			{
				// 首格的 SubShapeID：游程内体素值相同，取材质/法线用首格都是对的
				SubShapeIDCreator cell_id = voxel->EncodeSubShapeID(inSubShapeIDCreator1, uint(inX), uint(inY), uint(inZ));

				if (inRunLength == 1)
				{
					// 单格游程：还是内切球（胶囊的圆柱半高必须 > 0，单格没有"段"）
					CollisionDispatch::sCollideShapeVsShape(unit_sphere, inShape2, sphere_scale, inScale2, inFirstCellTransform, inCenterOfMassTransform2, cell_id, inSubShapeIDCreator2, inCollideShapeSettings, ioCollector, inShapeFilter);
				}
				else
				{
					// 一根胶囊覆盖整段：局部 Y 轴对齐到游程方向，中心落在游程中点
					const CapsuleShape *capsule = sGetUnitCapsule(inRunLength);
					Mat44 capsule_transform = Mat44::sRotationTranslation(Quat::sFromTo(Vec3::sAxisY(), inRunAxis), inRunCenter);
					CollisionDispatch::sCollideShapeVsShape(capsule, inShape2, sphere_scale, inScale2, capsule_transform, inCenterOfMassTransform2, cell_id, inSubShapeIDCreator2, inCollideShapeSettings, ioCollector, inShapeFilter);
				}
			});
		return;
	}

	voxel->sVisitSolidVoxelsInBox(local_box, inScale1, inCenterOfMassTransform1,
		[inShape2, inScale2, inCenterOfMassTransform2, voxel, &inSubShapeIDCreator1, &inSubShapeIDCreator2, &inCollideShapeSettings, &ioCollector, &inShapeFilter, unit_box, unit_sphere](int inX, int inY, int inZ, const Mat44 &inCellTransform, Vec3 inCellScale)
		{
			SubShapeIDCreator cell_id = voxel->EncodeSubShapeID(inSubShapeIDCreator1, uint(inX), uint(inY), uint(inZ));
			if (unit_sphere != nullptr)
			{
				// 内切球路径（非均匀缩放时的兜底，同 sCollideConvexVsVoxel 的说明）：每格 1 个点接触
				Vec3 sphere_scale = Vec3::sReplicate(inCellScale.Abs().ReduceMin());
				CollisionDispatch::sCollideShapeVsShape(unit_sphere, inShape2, sphere_scale, inScale2, inCellTransform, inCenterOfMassTransform2, cell_id, inSubShapeIDCreator2, inCollideShapeSettings, ioCollector, inShapeFilter);
			}
			else
			{
				CollisionDispatch::sCollideShapeVsShape(unit_box, inShape2, inCellScale, inScale2, inCellTransform, inCenterOfMassTransform2, cell_id, inSubShapeIDCreator2, inCollideShapeSettings, ioCollector, inShapeFilter);
			}
		});
}

// 一个实心格的 6 张格面里哪几张外露（bit 2k = 轴 k 的 + 面，bit 2k+1 = 轴 k 的 - 面）。
// sNormalMode == 3（融合模式）用它做路由：>= 2 面外露 = 棱格 / 角格（物体边缘）走球语义法线。
// ★ 这个量**只与这一格自己有关**，跟"对方在哪"无关 ⇒ 在窄相位里可以按格算一次、而不是按格对
//   算（一个遍历格最多有 8 个候选格对，原来要重复 8 遍同样的 6 次邻居查询）。
// 直接用公开的 IsSolid 写（等价于 IsVoxelFaceExposed = !IsSolid(邻居)，后者是 private）。
static inline int sGetExposedFaceMask(const VoxelShape *inShape, int inX, int inY, int inZ)
{
	int mask = 0;
	if (!inShape->IsSolid(inX + 1, inY, inZ)) mask |= 1;
	if (!inShape->IsSolid(inX - 1, inY, inZ)) mask |= 2;
	if (!inShape->IsSolid(inX, inY + 1, inZ)) mask |= 4;
	if (!inShape->IsSolid(inX, inY - 1, inZ)) mask |= 8;
	if (!inShape->IsSolid(inX, inY, inZ + 1)) mask |= 16;
	if (!inShape->IsSolid(inX, inY, inZ - 1)) mask |= 32;
	return mask;
}

void VoxelShape::sCollideVoxelGrids(const VoxelShape *inVoxel1, Vec3Arg inScale1, Mat44Arg inCenterOfMassTransform1, const SubShapeIDCreator &inSubShapeIDCreator1,
									const VoxelShape *inVoxel2, Vec3Arg inScale2, Mat44Arg inCenterOfMassTransform2, const SubShapeIDCreator &inSubShapeIDCreator2,
									const CollideShapeSettings &inCollideShapeSettings, CollideShapeCollector &ioCollector)
{
	// 任何一边没有实心格，就不可能接触
	if (inVoxel1->mNumSolidVoxels == 0 || inVoxel2->mNumSolidVoxels == 0)
		return;

	// ---- 1) 两个网格整体做一次 SAT，求出整片接触共用的分离轴 ----
	//
	// ★ 这一步是"能推开"的关键。如果省掉它、让每个格对各自算最小分离轴，那么在深穿透
	//   （两个立方体大面积重合，三个轴的重叠量相同）时每格挑到的轴都不一样，法线互相矛盾，
	//   求解器收到的净推进力互相抵消 —— 表现就是"两个方块插在一起、推不开"。
	//   用整块网格先定一根轴，所有格子共用它，接触集合就等价于"实心大盒撞实心大盒"。
	VoxelOBB obb1 = sGetGridOBB(inVoxel1->mSizeX, inVoxel1->mSizeY, inVoxel1->mSizeZ, inVoxel1->mVoxelSize, inScale1, inCenterOfMassTransform1);
	VoxelOBB obb2 = sGetGridOBB(inVoxel2->mSizeX, inVoxel2->mSizeY, inVoxel2->mSizeZ, inVoxel2->mVoxelSize, inScale2, inCenterOfMassTransform2);

	Vec3 axis_world;
	float grid_overlap = sObbMinSeparationAxis(obb1, obb2, axis_world, sLastGridOverlap[1]);
	sLastGridOverlap[0] = grid_overlap;				// 诊断用（见 VoxelShape.h）

	const float max_separation = inCollideShapeSettings.mMaxSeparationDistance;
	if (grid_overlap < -max_separation)
		return;										// 两个网格整体的最短分离都超过阈值了，不可能有接触

	// ---- 2) 两侧各圈一次"需要遍历的格子范围"，并选**遍历侧** ----
	//
	// 遍历范围 = 对方网格整体 OBB 在己方未缩放局部空间里的包围盒。保守一点没关系："多扫几格"
	// 只是浪费一点时间，而"漏格"会直接丢碰撞。外扩量按 mMaxSeparationDistance：靠得近但还没
	// 碰上的那一层也要参与（推测性接触），物体才会在真正插进去之前就收到接触力。
	//
	// 注意不要再额外加余量去抠"恰好相距 max_separation"这个边界：sGetCellRange 的边界规则是
	// "正好落在格子边界上的坐标归给上面那一格"（避免相邻格子重复计数），额外的余量会让边界格子
	// 同时被两侧算到，接触数翻倍、聚合面积也随之翻倍 —— 这会直接破坏"逐格接触的聚合 = 整面接触"
	// 这个契约（UnitTests 的 VoxelShapeVsVoxelShape 正是靠它校验的）。实测那个边界的差异完全被
	// Jolt 自己的 mPenetrationSlop（0.02）吸收掉了，对静止高度没有任何影响。
	//
	// ★ 遍历侧取"重叠区格数少"的那一侧：遍历成本 ∝ 遍历的格子数，而"大网格 vs 小方块"这种
	//   最常见的组合里，遍历小方块那一侧能把内层循环的候选查询量砍掉一个数量级以上。
	//   两侧格数一样（对称接触）时保持 shape1 侧，行为与旧版一致。
	AABox obb1_world = sObbWorldBox(obb1);
	AABox obb2_world = sObbWorldBox(obb2);
	obb1_world.ExpandBy(Vec3::sReplicate(max_separation));
	obb2_world.ExpandBy(Vec3::sReplicate(max_separation));

	AABox iter_box[2] =
	{
		sWorldBoxToVoxelLocal(obb2_world, inCenterOfMassTransform1, inScale1),	// 遍历 shape1 时的范围
		sWorldBoxToVoxelLocal(obb1_world, inCenterOfMassTransform2, inScale2)	// 遍历 shape2 时的范围
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

	int iterate_side = (has_range[0] && (!has_range[1] || count[0] <= count[1]))? 0 : 1;
	if (sIterateSideOverride != 0)						// 单元测试用：强制某个方向（"两个方向结果一致"的对照）
	{
		int forced = sIterateSideOverride - 1;
		if (has_range[forced])
			iterate_side = forced;
	}

	// ---- 3) 两侧的公共常量 ----
	//
	// 两侧需要的量是对称的，只是角色不同（遍历侧 / 候选侧），所以放同一张表里。
	struct GridSide
	{
		const VoxelShape *mVoxel;
		const SubShapeIDCreator *mIDCreator;
		Vec3 mScale;					// 形状缩放
		Mat44 mTransform;				// 质心变换
		Mat44 mToWorld;					// 单位盒（含格长）-> 世界（不含格心平移）
		float mRadius;					// 该侧格子在全局分离轴上的投影半径
		Vec3 mInvScale;					// 1 / mScale
		Vec3 mInvAbsScale;				// |1 / mScale|
		float mInvVoxelSize;			// 1 / 格长
		Vec3 mOrigin;					// 候选侧：格心世界坐标的基点
		Vec3 mStep[3];					// 候选侧：格心世界坐标的步进向量（下标 +1 走一格）
		Vec3 mFaceAxis[3];				// 单位化的格轴（= 格子的 6 张格面的法线方向），sNormalMode 2/3 用
		Vec3 mExpand;					// 候选侧：推测余量在该侧"格坐标系"里的分量
	};
	GridSide side[2];
	for (int i = 0; i < 2; ++i)
	{
		const VoxelShape *voxel = i == 0? inVoxel1 : inVoxel2;
		Vec3 scale = i == 0? inScale1 : inScale2;
		Mat44 transform = i == 0? inCenterOfMassTransform1 : inCenterOfMassTransform2;
		GridSide &s = side[i];

		s.mVoxel = voxel;
		s.mIDCreator = i == 0? &inSubShapeIDCreator1 : &inSubShapeIDCreator2;
		s.mScale = scale;
		s.mTransform = transform;
		s.mToWorld = transform * Mat44::sScale(scale);

		// 投影半径：同一侧所有格子的旋转都一样（格变换只改平移列），每查询算一次就够
		s.mRadius = sGetProjectionRadius(transform, scale * voxel->mVoxelSize, axis_world);

		s.mInvScale = Vec3(1.0f / scale.GetX(), 1.0f / scale.GetY(), 1.0f / scale.GetZ());
		s.mInvAbsScale = s.mInvScale.Abs();
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

		// 推测余量 e：全局分离轴映射到该侧格坐标系后的分量 × max_separation。只沿**全局轴**方向
		// 外扩，捞回轴向上"还差一点才碰上"的推测性接触（不扩的话物体会沉下去 max_separation 才停住）。
		// 垂直于轴的方向不外扩：那个方向上的缝隙本来就不该有接触（深度公式会给整格深度，全是噪声）。
		s.mExpand = (transform.Inversed().Multiply3x3(axis_world).Abs() * s.mInvAbsScale * s.mInvVoxelSize) * max_separation;
	}

	bool iterate_is_1 = iterate_side == 0;
	const GridSide &iter = side[iterate_side];			// 遍历侧：逐格枚举它
	const GridSide &cand = side[iterate_is_1? 1 : 0];	// 候选侧：被跨格枚举查询

	// 遍历侧的未缩放局部空间 -> 候选侧的未缩放局部空间
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
							+ inVoxel2->mVoxelSize * inScale2.Abs().ReduceMin()) + max_separation;
	float dist_limit = filter_dist / (cand.mVoxel->mVoxelSize * cand.mScale.Abs().ReduceMin());
	float dist_limit_sq = dist_limit * dist_limit;

	// ---- 4) 遍历**遍历侧**的实心格；每一格用跨格枚举找出**候选侧**里的候选实心格 ----
	//
	// 遍历顺序用 <true>：**由外往中心**。接触数撞到 cMaxVoxelContacts 截断时，先被保留的是重叠区
	// 外圈的格子 —— 剩下的接触仍然铺满整片区域的外围（形心和整片重合），不会像顺序截断那样只剩
	// 一侧（那会让物体被支撑在单边，发生倾斜/下陷，正是最早那个"下陷又被推上来"的成因链）。
	int num_contacts = 0;

	// 候选侧的常量（不随遍历格变）——提到 lambda 外面，省掉每格重复的取分量 + 数组构造
	const float ex[3] = { cand.mExpand.GetX(), cand.mExpand.GetY(), cand.mExpand.GetZ() };
	const int size[3] = { cand.mVoxel->mSizeX, cand.mVoxel->mSizeY, cand.mVoxel->mSizeZ };
	const int smin[3] = { cand.mVoxel->mSolidMin[0], cand.mVoxel->mSolidMin[1], cand.mVoxel->mSolidMin[2] };
	const int smax[3] = { cand.mVoxel->mSolidMax[0], cand.mVoxel->mSolidMax[1], cand.mVoxel->mSolidMax[2] };

	// ---- 4a) 一个遍历格的完整处理：候选侧格坐标 c、候选区间 lo/hi 已由调用方算好传进来 ----
	//
	// 为什么把前置量拆出来：**每格的固定开销**（格心 → c → 候选区间）跟这一格有没有候选、有几个
	// 候选都无关，是纯"每格一次"的成本；而候选循环平均每格只跑 1.3~1.8 次（实测）。所以这条路径
	// 的瓶颈是每格的固定开销，不是候选循环 —— 把 4 个遍历格的固定开销一起算（见 4b）才有意义。
	// 候选循环本身是数据相关的、长度不定，仍然逐格标量跑。
	auto run_cell = [&](int inX, int inY, int inZ, Vec3Arg inCenterIter, const float (&inCC)[3], const int (&inLo)[3], const int (&inHi)[3])
	{
		// 遍历侧这一格的 SubShapeID（只跟这一格有关，提到候选循环外面）
		SubShapeID id_iter = iter.mVoxel->EncodeSubShapeID(*iter.mIDCreator, uint(inX), uint(inY), uint(inZ)).GetID();

		// sSkipInteriorPairs：这一格是不是内部格（每格只查一次，不随候选重复）
		bool iter_is_interior = sSkipInteriorPairs && iter.mVoxel->IsVoxelInterior(inX, inY, inZ);

		// 融合模式（sNormalMode == 3）的路由判据 = owner 那一格的外露面 bitmask。它只与这一格
		// 自身有关、与对方在哪无关 ⇒ 当 owner 就是遍历侧这一格（iterate_is_1）时，**第一次用到
		// 时算一次**、后面所有候选格对直接复用，而不是写在格对循环里逐对重算（一个遍历格最多 8 个
		// 候选 ⇒ 同样的 6 次邻居查询被重复 8 遍）。懒算：这一格一个候选格对都没有时不付这个钱。
		// iterate_is_1 == false 时 owner 是候选侧那一格（每个格对各不同），没有可省的重复。
		int iter_exposed_mask = -1;			// -1 = 还没算（只对 iterate_is_1 有意义）

		// ---- 4) 逐格枚举：遍历侧的实心格 × 跨格枚举出的候选实心格 ----
		for (int z = inLo[2]; z <= inHi[2] && num_contacts < cMaxVoxelContacts; ++z)
			for (int y = inLo[1]; y <= inHi[1] && num_contacts < cMaxVoxelContacts; ++y)
				for (int x = inLo[0]; x <= inHi[0] && num_contacts < cMaxVoxelContacts; ++x)
				{
					if (!cand.mVoxel->IsSolid(x, y, z))
						continue;

					// 球近似筛选（候选侧格坐标系单位）：跨格集合比球宽（角上的格对），收回到球面内。
					// 直接用格下标算，不需要先算格心的世界坐标 —— 被刷掉的格对连 center_cand 都不用算。
					float dx = float(x) + 0.5f - inCC[0];
					float dy = float(y) + 0.5f - inCC[1];
					float dz = float(z) + 0.5f - inCC[2];
					if (dx * dx + dy * dy + dz * dz > dist_limit_sq)
						continue;

					// 内部格 x 内部格：两格都埋在自己实体的深处，跳过（见 sSkipInteriorPairs）。
					// 表面 x 实心、表面 x 表面都不受影响。
					if (iter_is_interior && cand.mVoxel->IsVoxelInterior(x, y, z))
						continue;

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
					if (sNormalMode == 1)
					{
						Vec3 to_shape2 = iterate_is_1? (center_cand - inCenterIter) : (inCenterIter - center_cand);
						float length = to_shape2.Length();
						if (length > 1.0e-6f)
							contact_axis = to_shape2 / length;
					}
					else if (sNormalMode == 2)
					{
						// 暴露面法线：在 **shape1 那一格**里挑一张"朝向 shape2、且没有被邻格挡住"的格面当法线。
						// 格面方向就是格子的 6 个格轴 ⇒ 法线落在格子局部轴上（与网格旋转无关），
						// 平面接触给出精确的格面法线，斜/曲面接触给出阶梯状局部法线（相邻格对的方向
						// 绕一圈在变 ⇒ 求解器聚合后等效于局部表面，这正是甜甜圈需要的"V 形槽"）。
						// 整格被埋住、没有任何朝向对方的暴露面时退回全局轴。
						// ★ 注意：曲面接触时这个法线**只是格轴方向**（阶梯），不是真正的斜面法线 ——
						//   它的作用是让相邻接触不再"全部同向"，代价是保留了体素的阶梯感。
						const VoxelShape *owner = iterate_is_1? iter.mVoxel : cand.mVoxel;
						const GridSide &owner_side = iterate_is_1? iter : cand;
						int owner_x = iterate_is_1? inX : x;
						int owner_y = iterate_is_1? inY : y;
						int owner_z = iterate_is_1? inZ : z;
						Vec3 to_shape2 = iterate_is_1? (center_cand - inCenterIter) : (inCenterIter - center_cand);

						// ★ 每根轴只需要**测一张面**，不必两张都测：
						//   把格轴 k 与 to_shape2 点一次得 d，两张格面的 dot 就是 ±d；而 best_dot 初值是 0
						//   ⇒ dot 为负的那张**永远赢不过**（它连 0 都过不了），只有 sign(d) 那一张可能是赢家。
						//   于是 6 次邻格查找 + 最多 6 次 dot 减半成 **最多 3 次查找 + 3 次 dot**，
						//   而且结果与"两张都测"**逐位相同**：face.Dot(to) = sign * (axis·to)（乘 ±1 在 IEEE
						//   下精确，求和取负也精确），比较用 |d| 与原来 dot 的值逐位一致，k 升序的平手规则不变。
						float best_dot = 0.0f;
						for (int k = 0; k < 3; ++k)
						{
							float d = owner_side.mFaceAxis[k].Dot(to_shape2);
							float ad = abs(d);
							if (ad > best_dot)
							{
								int dxf = k == 0? 1 : 0, dyf = k == 1? 1 : 0, dzf = k == 2? 1 : 0;
								int sign = d > 0.0f? 1 : -1;
								if (owner->IsVoxelFaceExposed(owner_x, owner_y, owner_z, sign * dxf, sign * dyf, sign * dzf))
								{
									best_dot = ad;
									contact_axis = owner_side.mFaceAxis[k] * float(sign);
								}
							}
						}
					}
					else if (sNormalMode == 3)
					{
						// 融合模式（"融化"）：按 **遍历侧那一格的外露面数量** 在 mode 1 / mode 2 之间分流。
						//   * 外露 >= 2 面（棱格、角格 = 物体的边缘）-> mode 1 的逐对格心连线（球语义）：
						//     方向可以是**任意角度**，不再量化到 6 个格轴 ⇒ 边缘保持圆润（像球体）。
						//   * 外露 < 2 面（平面格、以及整格埋住的深嵌格）-> mode 2 的暴露面法线：
						//     平面接触给出精确的格面法线（不会被 mode 1 算成 45° 斜推），整格埋住时退回
						//     全局轴（深嵌交错保持稳定）。
						// 动机：mode 1 翻车的地方是"对齐平面"（半格错位堆叠 0.707 ⇒ 斜推波浪）和"深嵌交错"，
						//   而这两种几何里的格子恰好几乎没有外露面；反过来 mode 1 更像球（方向贴斜面）的地方
						//   全是一圈外露 2~3 面的棱角格。按外露面数分流正好各取所长。
						// ★ 唯一弱点：判据作用在**遍历侧**的格子上（遍历侧 = 重叠区格数少的一方）。若遍历侧
						//   本身是一格厚的板（每格天生上下面都外露 = 2）⇒ 整块退化成 mode 1（实测沉 1.35 cm，
						//   见 `VoxelShapeNormalModeSimulation` 的 "thin iterated" 场景）。
						// ★ 成本与实现选型见下面注释与 `VoxelShapeNormalModeCost`。
						const VoxelShape *owner = iterate_is_1? iter.mVoxel : cand.mVoxel;
						const GridSide &owner_side = iterate_is_1? iter : cand;
						int owner_x = iterate_is_1? inX : x;
						int owner_y = iterate_is_1? inY : y;
						int owner_z = iterate_is_1? inZ : z;
						Vec3 to_shape2 = iterate_is_1? (center_cand - inCenterIter) : (inCenterIter - center_cand);

						// ★ 实现方式选型（交替多轮中位数实测过，别"优化"回去）：
						//   [现在] owner 的外露面 mask **按格算一次**（owner = 遍历侧那一格时提到候选循环
						//   外面，见 run_cell 开头），格对循环里只读 mask ⇒ 平面格（外露 = 1，绝大多数情况）
						//   每个**格** 6 次邻居查询 + 每**格对** 3 次 dot。
						//   [上一版] mask 写在格对循环里：平面格每个**格对** 6 次邻居查询 ⇒ 一个遍历格
						//   最多 8 个候选，同样的 6 次查询被重复 8 遍（这才是融合比 mode 2 慢的主因，
						//   不是"多查了 3 次邻居"）。
						//   [别再用] "一次遍历里每轴两张面都探"的单遍循环体：方向变成运行时 `sgn * dxf`
						//   加上数据相关的循环退出条件，编译器不展开 ⇒ 省不下来反被吃掉。
						// 路由：mask 有 >= 2 个 bit（棱格 / 角格 = 物体边缘）走 mode 1 的球语义；
						//       < 2（平面格 + 整格埋住）走 mode 2，且**不再查邻居**（mask 里已经有答案）。
						int exposed_mask;
						if (iterate_is_1)
						{
							if (iter_exposed_mask < 0)
								iter_exposed_mask = sGetExposedFaceMask(owner, owner_x, owner_y, owner_z);
							exposed_mask = iter_exposed_mask;
						}
						else
							exposed_mask = sGetExposedFaceMask(owner, owner_x, owner_y, owner_z);
						if (exposed_mask & (exposed_mask - 1))
						{
							// 边缘格（棱 / 角）-> mode 1：逐对格心连线（复用深度那行本来就要算的 Δc）
							float length = to_shape2.Length();
							if (length > 1.0e-6f)
								contact_axis = to_shape2 / length;
						}
						else
						{
							// 平面格 / 深嵌格 -> mode 2：每根轴只测 sign(d) 那一张面，外露与否直接查
							// 上面的 mask（轴 k 的 + 面在 bit 2k、- 面在 bit 2k+1）。
							float best_dot = 0.0f;
							for (int k = 0; k < 3; ++k)
							{
								float d = owner_side.mFaceAxis[k].Dot(to_shape2);
								float ad = abs(d);
								if (ad > best_dot)
								{
									int sign = d > 0.0f? 1 : -1;
									if (exposed_mask & (1 << (2 * k + (sign > 0? 0 : 1))))
									{
										best_dot = ad;
										contact_axis = owner_side.mFaceAxis[k] * float(sign);
									}
								}
							}
						}
					}

					// 深度：沿法线的投影重叠量（负值 = 推测性接触，最深到 -max_separation）
					float depth = radius_sum - abs((center_cand - inCenterIter).Dot(contact_axis));
					if (depth < -max_separation)
						continue;

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
					Vec3 point1 = center1 + contact_axis * radius_1;
					Vec3 point2 = center2 - contact_axis * radius_2;

					SubShapeID id_cand = cand.mVoxel->EncodeSubShapeID(*cand.mIDCreator, uint(x), uint(y), uint(z)).GetID();
					SubShapeID id1 = iterate_is_1? id_iter : id_cand;
					SubShapeID id2 = iterate_is_1? id_cand : id_iter;

					// 点接触：刻意**不**填 mShape1Face/mShape2Face。Jolt 在面为空时会退回用
					// mContactPointOn1/2 当单点接触（见 ManifoldBetweenTwoFaces），所以接触不会丢。
					CollideShapeResult result(point1, point2, contact_axis, depth, id1, id2, TransformedShape::sGetBodyID(ioCollector.GetContext()));
					ioCollector.AddHit(result);

					++num_contacts;
				}
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
	const Vec4 b_half = Vec4::sReplicate(0.5f);
	const Vec4 b_one = Vec4::sReplicate(1.0f);
	const Vec4 b_eps = Vec4::sReplicate(cCellEpsilon);
	const Vec4 b_ex0 = Vec4::sReplicate(ex[0]), b_ex1 = Vec4::sReplicate(ex[1]), b_ex2 = Vec4::sReplicate(ex[2]);
	const Vec4 b_lm0 = Vec4::sReplicate(float(max(0, smin[0]))), b_lm1 = Vec4::sReplicate(float(max(0, smin[1]))), b_lm2 = Vec4::sReplicate(float(max(0, smin[2])));
	const Vec4 b_hm0 = Vec4::sReplicate(float(min(size[0] - 1, smax[0]))), b_hm1 = Vec4::sReplicate(float(min(size[1] - 1, smax[1]))), b_hm2 = Vec4::sReplicate(float(min(size[2] - 1, smax[2])));

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
			fx[i] = float(buf_x[j]) + 0.5f;
			fy[i] = float(buf_y[j]) + 0.5f;
			fz[i] = float(buf_z[j]) + 0.5f;
			wx[i] = buf_c[j].GetX();
			wy[i] = buf_c[j].GetY();
			wz[i] = buf_c[j].GetZ();
		}

		Vec4 vfx(fx[0], fx[1], fx[2], fx[3]);
		Vec4 vfy(fy[0], fy[1], fy[2], fy[3]);
		Vec4 vfz(fz[0], fz[1], fz[2], fz[3]);

		// 候选侧格坐标 c：4 车道一次算完。结合顺序与标量式 (origin + t0) + t1 + t2 保持一致。
		Vec4 cx = b_org_x + vfx * b_s0x + vfy * b_s1x + vfz * b_s2x;
		Vec4 cy = b_org_y + vfx * b_s0y + vfy * b_s1y + vfz * b_s2y;
		Vec4 cz = b_org_z + vfx * b_s0z + vfy * b_s1z + vfz * b_s2z;

		// 候选下标区间：4 车道一次算完（floor/ceil + 夹到实心范围）。逐项与标量式一致。
		Vec4 lo0 = Vec4::sMax(sFloor4(cx - b_half - b_ex0 + b_eps), b_lm0);
		Vec4 hi0 = Vec4::sMin(sCeil4(cx + b_half + b_ex0 - b_eps) - b_one, b_hm0);
		Vec4 lo1 = Vec4::sMax(sFloor4(cy - b_half - b_ex1 + b_eps), b_lm1);
		Vec4 hi1 = Vec4::sMin(sCeil4(cy + b_half + b_ex1 - b_eps) - b_one, b_hm1);
		Vec4 lo2 = Vec4::sMax(sFloor4(cz - b_half - b_ex2 + b_eps), b_lm2);
		Vec4 hi2 = Vec4::sMin(sCeil4(cz + b_half + b_ex2 - b_eps) - b_one, b_hm2);

		Float4 a_lo0, a_hi0, a_lo1, a_hi1, a_lo2, a_hi2, a_cx, a_cy, a_cz;
		lo0.StoreFloat4(&a_lo0); hi0.StoreFloat4(&a_hi0);
		lo1.StoreFloat4(&a_lo1); hi1.StoreFloat4(&a_hi1);
		lo2.StoreFloat4(&a_lo2); hi2.StoreFloat4(&a_hi2);
		cx.StoreFloat4(&a_cx); cy.StoreFloat4(&a_cy); cz.StoreFloat4(&a_cz);

		for (int lane = 0; lane < inN && num_contacts < cMaxVoxelContacts; ++lane)
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

	iter.mVoxel->sVisitSolidVoxelsInBox<true>(iter_box[iterate_side], iter.mScale, iter.mTransform,
		[&](int inX, int inY, int inZ, const Mat44 &inCellTransform, Vec3)
		{
			if (num_contacts >= cMaxVoxelContacts)
				return;

			buf_x[nbuf] = inX;
			buf_y[nbuf] = inY;
			buf_z[nbuf] = inZ;
			buf_c[nbuf] = inCellTransform.GetTranslation();
			if (++nbuf == 4)
			{
				flush_batch(4);
				nbuf = 0;
			}
		});

	if (nbuf > 0)
		flush_batch(nbuf);
}

void VoxelShape::sCastConvexVsVoxel(const ShapeCast &inShapeCast, const ShapeCastSettings &inShapeCastSettings, const Shape *inShape, Vec3Arg inScale, const ShapeFilter &inShapeFilter, Mat44Arg inCenterOfMassTransform2, const SubShapeIDCreator &inSubShapeIDCreator1, const SubShapeIDCreator &inSubShapeIDCreator2, CastShapeCollector &ioCollector)
{
	// 这个函数注册在 (任意形状, 体素) 上：inShape 一定是体素形状。
	// inShapeCast 已经表达在体素形状的局部空间（含缩放）里，扫掠范围 = 起点包围盒 ∪ 终点包围盒
	// （cast 过程中形状只平移不旋转，所以起点和终点的包围盒就够了）。
	const VoxelShape *voxel = static_cast<const VoxelShape *>(inShape);

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

	const BoxShape *unit_box = sGetUnitBox();
	voxel->sVisitSolidVoxelsInBox(local_box, inScale, inCenterOfMassTransform2,
		[&inShapeCast, &inShapeCastSettings, &inShapeFilter, &inSubShapeIDCreator1, voxel, inScale, &inSubShapeIDCreator2, unit_box, &ioCollector](int inX, int inY, int inZ, const Mat44 &inCellTransform, Vec3 inCellScale)
		{
			// 把该格搬到原点，于是 shape cast 也要反向平移格心。
			//
			// ★ 这里的平移必须作用在"体素形状的坐标系"里，而不是 cast 自己的坐标系里：
			//   cast 可能是带旋转的，如果按 cast 自己的坐标系平移，格子就会落错位置
			//   （斜着投出去的射线会命中不存在的格子，或者漏掉真正的格子）。
			Vec3 cell_center_in_cast_space = voxel->GetVoxelCenter(inX, inY, inZ) * inScale;
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
