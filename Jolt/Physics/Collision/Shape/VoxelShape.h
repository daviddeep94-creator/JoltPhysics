// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/Collision/Shape/SubShapeID.h>
#include <Jolt/Physics/Collision/PhysicsMaterial.h>
#include <Jolt/Core/Array.h>
#include <Jolt/Core/Atomics.h>
#include <string.h>

#ifdef JPH_DEBUG_RENDERER
	#include <Jolt/Renderer/DebugRenderer.h>
#endif // JPH_DEBUG_RENDERER

JPH_NAMESPACE_BEGIN

class BoxShape;
class CapsuleShape;
class CollideShapeSettings;
class SphereShape;

/// 构造 VoxelShape 的设置对象（"未烹调"的体素形状描述）。
///
/// 体素形状是一个规则的立方体素网格。形状**不拷贝**也**不拥有**体素数据：在形状存活期间，调用方必须保证这块
/// 缓冲区一直有效、地址不变、长度不变。
///
/// ★ **存储格式 = 占用位图，1 bit / 体素**（取代早先的 1 字节/体素；字节里的"材质"已经移到按 chunk 的材质图上）。
///   数据是一串 uint64：**一个 uint64 = 一个 4x4x4 子块 = 64 格**，正好一位一格。
///   - 位序：`bit = (x&3) | ((y&3)<<2) | ((z&3)<<4)`（= sGetSubChunkBit，x 占最低位）
///   - 字序：`word = (x>>2) + (y>>2) * texelsX + (z>>2) * texelsX * texelsY`，texelsX = ceil(sizeX / 4)
///   - 一个 chunk（8x8x8）正好是 2x2x2 个这样的字。
///   "一个字 = 一个 4x4x4 子块"这个粒度不是随手选的，它是同一个东西的三重身份：一个 64 位机器字 /
///   窄相位跳过空区的原子单位 / **渲染侧那张 3D 纹理的一个像素**（4 个 16 位通道凑一个 64 位像素）。
///   物理和渲染因此可以直接共用同一块缓冲区，见下面的硬约束 1。
///   ⚠ 材质不占体素位：材质是每个 8x8x8 chunk 一个 16 位编号，单独一张图。
///
/// 形状的局部原点位于网格中心，所以局部包围盒的尺寸是
/// (sizeX * voxelSize, sizeY * voxelSize, sizeZ * voxelSize)，即局部坐标范围是 ±0.5 * size * voxelSize。
///
/// 因为每次碰撞查询都直接读取体素数据，所以可以在刚体存活期间就地增删体素：既不需要重建形状，也不需要重建
/// 碰撞体，而重建碰撞体通常是可破坏体素世界里最贵的一步。修改数据之后调用 VoxelShape::Recalculate() 刷新缓存
/// 的质量属性；如果包围盒发生了变化，再用 BodyInterface::NotifyShapeChanged 通知刚体。
///
/// 注意：体素形状不支持二进制序列化 —— 它不拥有体素数据，把"数据指针"存进流里没有意义。
class JPH_EXPORT VoxelShapeSettings final : public ShapeSettings
{
public:
	JPH_OVERRIDE_NEW_DELETE

	/// 默认构造函数
							VoxelShapeSettings() = default;

	/// 构造一个 sizeX * sizeY * sizeZ 个体素、每格 voxelSize 米、占用位图取自 inOccupancy 的设置
							VoxelShapeSettings(int inSizeX, int inSizeY, int inSizeZ, float inVoxelSize, const uint64 *inOccupancy, float inDensity = 1000.0f) :
		mSizeX(inSizeX),
		mSizeY(inSizeY),
		mSizeZ(inSizeZ),
		mVoxelSize(inVoxelSize),
		mOccupancy(inOccupancy),
		mDensity(inDensity)
	{
	}

	/// 见 ShapeSettings::Create
	virtual ShapeResult		Create() const override;

	int						mSizeX = 0;						///< 沿 X 轴的体素数量
	int						mSizeY = 0;						///< 沿 Y 轴的体素数量
	int						mSizeZ = 0;						///< 沿 Z 轴的体素数量
	float					mVoxelSize = 0.0f;				///< 单个体素的边长（米）
	const uint64 *			mOccupancy = nullptr;			///< 占用位图：1 bit/体素，一个 uint64 = 一个 4x4x4 子块（不拥有所有权）
	float					mDensity = 1000.0f;				///< 密度（kg/m^3），用来算质量和转动惯量
	RefConst<PhysicsMaterial> mMaterial;					///< 碰撞材质
};

/// 由立方体素网格构成的碰撞体。
///
/// **这是一个非凸形状**：它注册成自定义形状 EShapeType::User1 / EShapeSubType::User1（而不是"用户自定义凸
/// 形状" UserConvex1）。窄相位按**逐格**的方式求接触，所以凹的体素数据会按"画出来的样子"参与碰撞 ——
/// 墙上的窗户可以穿过去、挖出来的隧道是真正的隧道、倒塌的楼房会散架而不是变成一个实心大块。
/// 如果对体素求凸包（上一版实现），洞会被填平，上述行为全部失效。
///
/// 关键实现点（这三条是硬约束，改代码时不要破坏）：
///
/// 1. **绝不拷贝体素数据**。形状只持有调用方那张占用位图的裸指针（mOccupancy），不复制、不拥有。
///    Unity 侧渲染和物理共用同一份数据；一旦复制，内存直接翻倍 —— 对大世界来说这是不可接受的。
///    Recalculate() 算出来的只是一小撮统计量，不是数据副本。
/// 2. **绝不为单个体素创建形状对象**。整个类里没有任何一处 `new BoxShape` 是按体素来的。
///    - 体素 vs 体素走的是一条**手写的窄相位**（sCollideVoxelGrids）：直接比较两个格子盒的重叠，
///      不查碰撞分发表、不构造任何形状对象、不做 GJK/EPA。
///    - 其他形状（Sphere / Box / Capsule / Mesh / ...）撞上来时，所有格子共用**同一个**进程级单位盒子
///      （sGetUnitBox()，边长 1、中心在原点、凸半径 0）或进程级单位球（sGetUnitSphere()，sUseSphereContacts
///      开启时，半径 0.5 的内切球），每格只是"缩放 + 平移"一下，没有任何一格拥有自己的形状对象。
/// 3. **只算重叠部分的体素**。窄相位只遍历"两个形状包围盒重叠区"里的实心格（sVisitSolidVoxelsInBox），
///    用两级跳过空区域：一个 4x4x4 子块（= 占用位图里的一个字）为空就一次跳过 64 格，一行里连续的
///    4 格（= 字里的一个 nibble）为空再跳 4 格。**这两级读的就是占用位图本身**，没有第二份索引
///    （早先的 mChunkMasks 已删除）⇒ 不会出现过期索引导致漏碰撞。所以与一面墙碰撞的开销和地图多大无关，
///    只与重叠区里的实心格数量有关（见 UnitTests 里的 VoxelShapeQueryScaling）。
///
/// 另外两个接口约定：
///
/// - **SubShapeID 编码体素坐标**（见 EncodeSubShapeID / DecodeSubShapeID），破坏系统据此定位被击中的格子。
/// - **质量属性是缓存值**。碰撞查询读的是实时体素数据，但质量/惯量只在 Recalculate() 时才重算，所以
///   就地删体素后刚体的行为会立刻变化，而质量要等下一次 Recalculate() 才更新。
class PhysicsSystem;									///< 前向声明：sInstallMaterialCombine 的参数（避免头文件依赖整个 PhysicsSystem）

/// 体素材质：带**摩擦 / 弹性**的 PhysicsMaterial 子类。
///
/// ★ 为什么必须自己定义：Jolt 的 PhysicsMaterial 基类**不含**摩擦和弹性 —— 那两个量在
///   Body 上（Body::GetFriction / GetRestitution），而一个地形区块是**一个** Body ⇒
///   光靠 Body 只能给整块地形一个摩擦值，做不到"冰面滑、沙地涩"。
///   解法：把摩擦/弹性挂在材质上，再由 PhysicsSystem 的 CombineFriction /
///   CombineRestitution 回调读取 —— 那两个回调**能拿到 SubShapeID**，而体素形状的
///   SubShapeID 里就编码了体素坐标（见 VoxelShape::sInstallMaterialCombine）。
class VoxelMaterial final : public PhysicsMaterial
{
public:
	JPH_OVERRIDE_NEW_DELETE

	float					mFriction = 0.5f;				///< 摩擦系数（0 = 绝对光滑）
	float					mRestitution = 0.0f;				///< 弹性（0 = 完全不弹）
};

/// 逐体素材质的档位表（与 C# 侧 VxlMaterialStore 的 kWidth / kTableCap / kTableWords 严格一致）
///   code 0 = 1 种（无编号数组） / 1 = 1bit / 2 = 2bit / 3 = 4bit / 4 = 16bit 直存（无索引表）
static constexpr uint		kVoxelMatWidth[5] = { 0, 1, 2, 4, 16 };
static constexpr uint		kVoxelMatTableCap[5] = { 1, 2, 4, 16, 0 };
static constexpr uint		kVoxelMatTableWords[5] = { 1, 1, 1, 4, 0 };

class JPH_EXPORT VoxelShape final : public Shape
{
public:
	JPH_OVERRIDE_NEW_DELETE

	/// 默认构造函数（供 ShapeFunctions::mConstruct 使用，之后必须调用 Recalculate()）
							VoxelShape() :
		Shape(EShapeType::User1, EShapeSubType::User1)
	{
	}

	/// 从设置构造
							VoxelShape(const VoxelShapeSettings &inSettings, ShapeResult &outResult);

	/// 便捷构造函数：直接给出网格尺寸、体素边长和占用位图
							VoxelShape(int inSizeX, int inSizeY, int inSizeZ, float inVoxelSize, const uint64 *inOccupancy, float inDensity = 1000.0f);

	/// 沿 X / Y / Z 轴的体素数量
	int						GetSizeX() const						{ return mSizeX; }
	int						GetSizeY() const						{ return mSizeY; }
	int						GetSizeZ() const						{ return mSizeZ; }

	/// 单个体素的边长（米）
	float					GetVoxelSize() const					{ return mVoxelSize; }

	/// 占用位图（不拥有所有权）：1 bit/体素，**一个 uint64 = 一个 4x4x4 子块**
	const uint64 *			GetOccupancy() const					{ return mOccupancy; }

	/// 占用位图沿 X / Y 轴的 texel 数（= ceil(size / 4)）。渲染侧那张 3D 纹理的分辨率就是这两个值。
	int						GetTexelsX() const						{ return mTexelsX; }
	int						GetTexelsY() const						{ return mTexelsY; }

	/// 体素数据的版本号：每次 Recalculate() 都换一个新的**全局唯一**值。除了让缓存的调试几何失效，
	/// 它也是边缘格缓存键的一部分（见 sEdgeCacheEnabled）—— 数据改过以后旧条目自动对不上。
	uint					GetVoxelDataVersion() const				{ return mVersion; }

	/// 格 (x, y, z) 在占用位图里的**字下标**（= 它所属 4x4x4 子块的编号）
	inline int				GetTexelIndex(int inX, int inY, int inZ) const
	{
		return (inX >> cSubChunkShift) + (inY >> cSubChunkShift) * mTexelsX + (inZ >> cSubChunkShift) * mTexelPlane;
	}

	/// 读取格 (x, y, z) 所在的整个 64 位字（= 它的 4x4x4 子块）。越界一律返回 0（= 视为空）。
	/// 遍历时用它一次拿到 64 格的占用情况，比逐格 IsSolid 便宜得多。
	inline uint64			GetWord(int inX, int inY, int inZ) const
	{
		if (inX < 0 || inX >= mSizeX || inY < 0 || inY >= mSizeY || inZ < 0 || inZ >= mSizeZ)
			return 0;
		return mOccupancy[GetTexelIndex(inX, inY, inZ)];
	}

	/// 该格是否为实心（越界一律视为空）
	inline bool				IsSolid(int inX, int inY, int inZ) const
	{
		return (GetWord(inX, inY, inZ) >> sGetSubChunkBit(inX, inY, inZ)) & 1;
	}

	/// 读取一格：true = 实心，false = 空。越界一律返回 false。（等价于 IsSolid，名字更直白）
	inline bool				GetSolid(int inX, int inY, int inZ) const
	{
		return IsSolid(inX, inY, inZ);
	}

	/// 设置一个格：inSolid = true 实心 / false 空。越界坐标会被忽略。
	///
	/// 这会**立即**改变碰撞几何 —— 窄相位每次查询都直接读占用位图，所以写入之后不需要重建形状或刚体。
	/// 但它**不会**刷新任何派生缓存（实心格计数、实心包围范围、质量属性）：
	/// 改完体素要刷新时，用 UpdateSolid()（增量，O(1)，推荐）或 Recalculate()（全量，O(网格体积)）。
	///
	/// 形状不拥有体素数据，所以这里写的是调用方的缓冲区（指针成员是 const 的，内部做一次 const_cast）。
	/// 这里也是 const 成员，与 Recalculate() 保持一致的语义：破坏系统可以在 const 形状上改体素。
	inline void				SetSolid(int inX, int inY, int inZ, bool inSolid) const
	{
		if (inX < 0 || inX >= mSizeX || inY < 0 || inY >= mSizeY || inZ < 0 || inZ >= mSizeZ)
			return;
		uint64 &word = const_cast<uint64 *>(mOccupancy)[GetTexelIndex(inX, inY, inZ)];
		uint64 bit = uint64(1) << sGetSubChunkBit(inX, inY, inZ);
		if (inSolid)
			word |= bit;
		else
			word &= ~bit;
	}

	/// 体素中心相对于形状原点（= 网格中心）的坐标
	inline Vec3				GetVoxelCenter(int inX, int inY, int inZ) const
	{
		return Vec3(
			(inX + 0.5f) * mVoxelSize - mHalfExtent.GetX(),
			(inY + 0.5f) * mVoxelSize - mHalfExtent.GetY(),
			(inZ + 0.5f) * mVoxelSize - mHalfExtent.GetZ());
	}

	/// 实心体素的数量（缓存值，只有 Recalculate() 之后才是最新的）
	/// 实心体素数量。**第一次调用会触发惰性统计**（见 EnsureDerived），静态形状通常没人问，所以永远不会算。
	uint					GetNumSolidVoxels() const				{ EnsureDerived(); return mNumSolidVoxels; }

	/// 实心格包围范围的轴下标。**没做过惰性统计时返回整张网格**（不触发扫描）—— 它只用于剪枝，宽一点只影响速度。
	int						GetSolidMin(int inAxis) const			{ return mDerivedValid? mSolidMin[inAxis] : 0; }
	int						GetSolidMax(int inAxis) const			{ return mDerivedValid? mSolidMax[inAxis] : GetAxisSize(inAxis) - 1; }

	/// **实心体素的质量质心**（重心），相对形状原点（= 网格中心）的坐标；一个实心格都没有时返回 0。
	///
	/// ★★ 与 Shape::GetCenterOfMass() 的区别（很重要，别当成同一个东西）：
	///   - Shape::GetCenterOfMass() 是 Jolt 的"形状声明的重心偏移"。VoxelShape 覆写它并返回
	///     mCenterOfMass（默认 Vec3::sZero() = 网格中心，可由 SetCenterOfMass() 改），而
	///     mMassProperties.mInertia 是**绕网格中心**算的（Recalculate() 里用 GetVoxelCenter() 做平行轴定理）。
	///   - 本函数返回的是**真实的质量质心**：所有实心格（体积相同 ⇒ 质量相同）格心坐标的平均值。
	///     几何不对称的模型（一侧实、一侧空）会明显偏离网格中心。
	///
	/// ★ 这个值**不参与任何物理计算**：物理重心用的是 SetCenterOfMass() 设定的值（默认 0 = 网格中心）。
	///   它的用途是**可视化 / 分析** —— 例如在编辑器里把质心画出来，作为 SetCenterOfMass() 的入参参考。
	/// ★★ 因此**不在 Recalculate() 里逐格累加**（那是每格一次 GetVoxelCenter + 3 次 double 加法；
	///   128³ / 78 万实心格的块实测让 Recalculate 慢 ~10%），而物理根本不用这个值。
	///   改成**按需计算 + 缓存**：第一次调用时顺带做一次惰性统计（EnsureDerived），之后直接返回缓存，
	///   直到 Recalculate() / UpdateSolid() 把它作废。
	/// ★ 累加量（Σ 格心坐标）与实心格数、惯量、实心包围范围**在同一趟扫描里**一起算完（见 EnsureDerived）：
	///   那里的内层循环已经对"每条轴上每个取值"做了 popcount，顺手乘上坐标即可，边际成本为 0；
	///   而旧实现在这里另起了一遍三重循环逐格 IsSolid（128³ 要问 200 万次，其中绝大多数是空格）。
	///   增量编辑（UpdateSolid）沿用与惯量相同的差值维护。
	Vec3					GetMassCenter() const;

	/// 让所有派生缓存失效（实心体素数量、实心包围范围、质量属性、质心累加、调试几何、数据版本）。
	/// 在就地修改体素数据之后必须调用。
	///
	/// ★★ 它**自己不扫描任何体素**：派生量改为惰性统计（EnsureDerived()），只有真正被问到才扫一遍。
	///   地形区块是静态体，质量和惯量不参与求解 ⇒ 建区块时这里只走一次几何常量，成本与体积无关。
	///
	/// 注意这个函数是 **const** 的：碰撞查询可以在 const 形状上刷新缓存，所以破坏系统不需要拿到非 const 形状。
	/// 为此下面所有缓存成员都是 mutable。
	void					Recalculate() const;

	/// 就地修改**一个**格，并**增量**维护所有派生缓存（实心格计数、实心包围范围、质量属性、
	/// 数据版本）—— 破坏 / 放置体素应该用这个，而不是 SetSolid() + Recalculate()。
	///
	/// 代价与网格尺寸**无关**，而 Recalculate() 是 O(网格体积) 的线性扫描。占用位图那一位就是
	/// "这一格是否实心"，与邻居无关，所以是 O(1) 位操作；其余只有按差值更新的质量属性
	/// 和只在实心时才扩的包围范围。
	///
	/// 与 SetSolid() / Recalculate() 的分工：
	///   - 只改位用 SetSolid()（或直接写缓冲区），**不刷新缓存**；
	///   - 改完要刷新：UpdateSolid()（增量，推荐）或 Recalculate()（全量，精确）。
	///
	/// ★ 唯一与 Recalculate() 不同的地方：实心包围范围（mSolidMin/mSolidMax）**只扩不缩** —— 删除体素
	///   造成的收缩要等下一次 Recalculate() 才精确。它是遍历范围的**保守上界**（只会让遍历多走几格，
	///   不会漏格、不影响任何查询结果）。全部实心格被删光时会重置成与 Recalculate() 相同的空状态。
	/// ★ 实心性没变（写入的值与原来相同）时直接返回，不做任何缓存更新，也不换数据版本。
	/// ★ 线程：与 Recalculate() 一样，**不要在别的线程正在查询同一个形状时调用**。
	void					UpdateSolid(int inX, int inY, int inZ, bool inSolid) const;

	/// UpdateSolids() 的一项编辑
	struct SolidEdit
	{
		int					mX;
		int					mY;
		int					mZ;
		bool				mSolid;
	};

	/// 批量版 UpdateSolid()：逐项做增量维护，但**只在最后统一换一次数据版本**（边缘格缓存的失效与
	/// 调试几何失效都只做一次）。破坏一片体素时用这个比逐格调用 UpdateSolid() 更省。
	/// 同一个格子在一批里出现多次是允许的（按顺序依次生效）。
	void					UpdateSolids(const SolidEdit *inEdits, int inCount) const;

	/// 把一个体素坐标编码进 SubShapeID（低若干位是 X，然后是 Y，最后是 Z）。
	/// 每个轴需要多少位由网格尺寸决定（见 sGetBitsForSize）。
	SubShapeIDCreator		EncodeSubShapeID(const SubShapeIDCreator &inCreator, uint inX, uint inY, uint inZ) const
	{
		return inCreator.PushID(inX, mBitsX).PushID(inY, mBitsY).PushID(inZ, mBitsZ);
	}

	/// EncodeSubShapeID 的逆运算
	void					DecodeSubShapeID(const SubShapeID &inSubShapeID, uint &outX, uint &outY, uint &outZ) const
	{
		SubShapeID remainder = inSubShapeID;
		SubShapeID shifted;
		outX = remainder.PopID(mBitsX, shifted); remainder = shifted;
		outY = remainder.PopID(mBitsY, shifted); remainder = shifted;
		outZ = remainder.PopID(mBitsZ, shifted);
	}

	// 见 Shape::GetLocalBounds
	virtual AABox			GetLocalBounds() const override
	{
		return AABox(-mHalfExtent, mHalfExtent);
	}

	// 见 Shape::GetInnerRadius
	virtual float			GetInnerRadius() const override
	{
		return 0.0f;
	}

	// 见 Shape::GetMassProperties
	virtual MassProperties	GetMassProperties() const override
	{
		// 质量 / 惯量是惰性统计的：静态形状（地形区块）永远不会走到这里，也就不付任何代价
		EnsureDerived();
		return mMassProperties;
	}

	// 见 Shape::GetCenterOfMass
	/// 物理质心：由 SetCenterOfMass() 显式设定，默认 Vec3::sZero() = 网格中心（形状原点）。
	/// 体素形状不拿逐格统计出的真实质心（GetMassCenter）当物理质心，那个只用于可视化 / 分析。
	virtual Vec3			GetCenterOfMass() const override
	{
		return mCenterOfMass;
	}

	/// 设置物理质心（相对形状原点 = 网格中心）。动态体素刚体（碎块）建体前调用一次：
	/// 传 `Vec3::sZero()` = 质心取网格中心（碎块网格紧贴实心包围盒 ⇒ 也是包围盒中心）。
	/// ★ 本形状的惯量缓存（mInertiaDiag）是**绕网格中心**算的，这里只改质心声明值，
	///   所以只应传 0；传非零需要另行平移惯量。
	void					SetCenterOfMass(Vec3Arg inCenterOfMass) const
	{
		mCenterOfMass = inCenterOfMass;
	}

	/// 声明"本网格在 ±X / ±Z 四个水平面方向上是**延续到邻区块的**"（地形区块专用）。
	///
	/// ★★ 为什么需要它：地形是一大片**相邻 VoxelShape 拼**出来的，每个区块自己并不知道隔壁有什么。
	///   窄相位把这样的一个区块当成独立的 8 m 大盒来算 SAT 轴（见 VoxelShape.cpp 的 sObbMinSeparationAxis），
	///   而"最小行程"总是把小物体推向**离它最近的那个面** —— 物体停在区块侧壁附近时，那个面就是区块的
	///   假边界。于是：
	///     * 静置时 ⇒ 整片接触的法线是纯水平的 ⇒ 物体被横向推、跨过缝进隔壁区块又被反向推回（左右摇滚）；
	///     * 埋进地形时 ⇒ 应该往上顶，却给了水平轴 ⇒ 净推力互相抵消，物体卡在穿透位置抖动（"推不开"）。
	///   标了延续之后，±X / ±Z 不再是合法出口（那边还有地形），候选只剩垂直方向 ⇒ 两种情况都回到正解。
	///
	/// 这是一个**形状自身的固有属性**（"这块体素是一片连续地形的一部分"），跟 shared_ptr 无关，
	/// 所以 setter 是 const 且要 ++mVersion 让一切按版本号缓存的东西失效。
	void					SetBorderContinues(bool inContinues) const
	{
		mBorderContinues = inContinues;
		++mVersion;
	}

	bool					GetBorderContinues() const
	{
		return mBorderContinues;
	}

	/// mBorderContinues 对应的轴掩码：bit k = 第 k 条网格轴（0=X, 1=Y, 2=Z）那边是延续的。
	/// Y 方向**不算延续** —— 地形在垂直方向有真实边界（上方是天空、下方是世界底），
	/// 而且它必须是候选，否则这种 pair 一根候选轴都不剩。
	uint					GetContinuousMask() const
	{
		return mBorderContinues? (1u | 4u) : 0u;
	}

	// 见 Shape::GetSubShapeIDBitsRecursive
	virtual uint			GetSubShapeIDBitsRecursive() const override
	{
		return mBitsX + mBitsY + mBitsZ;
	}

	// ---- 逐体素材质 ----
	// ★★ **形状自己持有一份拷贝**（与 mOccupancy 那条"借用"约定不同，故意的）：
	//   调用方那份（Unity 的 VxlMaterialStore）会因为**扩容 / 压实**而换数组或重排，
	//   而物理读材质发生在求解 job 里（多线程）。借裸指针就要求"调用方在任何时刻改数据
	//   都必须精确地同步给形状"，这个前提在编辑路径上根本保证不了 —— 实际后果是
	//   use-after-free（GC 回收掉已解 pin 的旧数组）⇒ 进程直接消失，没有托管异常可抓。
	//   拷进来之后形状只碰自己的内存，调用方随便改、随便丢，**不可能再读到野地址**。
	//   代价 = 每个形状多一份 header + 记录池（实测 ~5 KB / 64×56×64 区块，
	//   远小于同区块的占用位图 NativeArray），换来的是崩溃类问题整类消失。
	// ★ 位序与 C# 的 VxlMaterialStore、shader 的 VxlMaterialAt **完全一致**。
	// ★ 两份数组都非空时 GetMaterial() 才走逐格；否则退回整形状一个 mMaterial。
	// ★★ 长度（headerCount / dataWords）一并拷下来做越界检查：一旦两侧布局有任何错位，
	//    退化成"这一格取不到材质 ⇒ 返回 0 ⇒ 用形状默认材质"，**永不越界读**。
	//    调用时机约定：**只在物理步进之外**（主线程）调用，与 PhysicsSystem::Update 不重叠。
	void					SetMaterialData(const uint32 *inHeader, uint inHeaderCount,
											const uint64 *inData, uint inDataWords)
	{
		if (inHeader == nullptr || inData == nullptr || inHeaderCount == 0 || inDataWords == 0)
		{
			mMaterialHeader.clear();
			mMaterialData.clear();
			return;
		}
		// resize 只改长度、**不缩容量**（见 Jolt Array::reserve）⇒ 稳定后不再分配。
		mMaterialHeader.resize(inHeaderCount);
		memcpy(mMaterialHeader.data(), inHeader, size_t(inHeaderCount) * sizeof(uint32));
		mMaterialData.resize(inDataWords);
		memcpy(mMaterialData.data(), inData, size_t(inDataWords) * sizeof(uint64));
	}

	/// 是否挂了逐体素材质数据
	bool					HasMaterialData() const						{ return !mMaterialHeader.empty() && !mMaterialData.empty(); }

	/// 取格 (x, y, z) 的**全局材质编号**（两级：编号位流 → 索引表 → 方块编号）。
	/// 没有材质数据、越界、或该格没材质 ⇒ 返回 0（= 用形状的默认材质）。
	inline uint				GetMaterialId(int inX, int inY, int inZ) const
	{
		if (mMaterialHeader.empty() || mMaterialData.empty())
			return 0;
		// ★ 每次进来取一次本地指针/长度（都是自己的内存 ⇒ 永不过期）
		const uint32 *header = mMaterialHeader.data();
		const uint64 *data = mMaterialData.data();
		uint header_count = uint(mMaterialHeader.size());
		uint data_words = uint(mMaterialData.size());
		if (inX < 0 || inX >= mSizeX || inY < 0 || inY >= mSizeY || inZ < 0 || inZ >= mSizeZ)
			return 0;

		// 8³ chunk 下标（与 C# 的 ChunkIndexOf 同一公式）
		int chunks_x = (mSizeX + 7) >> 3;
		int chunks_y = (mSizeY + 7) >> 3;
		int ci = (inX >> 3) + (inY >> 3) * chunks_x + (inZ >> 3) * chunks_x * chunks_y;

		// ★ header 越界 / 档位非法（code 5..7）⇒ 取不到材质，返回 0。**不读内存**
		if (uint(ci) >= header_count)
			return 0;

		uint32 h = header[ci];
		uint base_w = h & 0x000FFFFFu;
		uint code = (h >> 20) & 7u;
		if (code > 4)
			return 0;

		// 索引表第 i 项（16 bit）。★ 每次读之前查字数：base_w 来自 header，可能指向池外
		// （两侧布局错位的唯一症状就是它）⇒ 越界一律当"没材质"，绝不解引用。
		auto table = [data, data_words, base_w](uint inIndex) -> uint
		{
			uint bit = base_w * 64u + inIndex * 16u;
			uint wi = bit >> 6;
			if (wi >= data_words)
				return 0;
			return uint((data[wi] >> (bit & 63)) & 0xFFFFu);
		};

		// code 0：没有编号数组，整块就是一种材质 ⇒ 直接读表[0]
		if (code == 0)
			return table(0);

		uint w = kVoxelMatWidth[code];
		uint tab_w = kVoxelMatTableWords[code];

		// chunk 内序号：x 占最低 3 位（与 C# 的 LocalIndex 一致）
		uint k = uint(inX & 7) | (uint(inY & 7) << 3) | (uint(inZ & 7) << 6);
		uint bit = k * w;

		// 位宽恒为 2 的幂且 ≤ 16 ⇒ 一个编号**永不跨 16 位通道**，也不会跨 64 位字
		uint wi = base_w + tab_w + (bit >> 6);
		if (wi >= data_words)
			return 0;
		uint num = uint((data[wi] >> (bit & 63)) & ((uint64(1) << w) - 1));

		if (code == 4)									// 直存：编号就是全局索引
			return num;
		if (num >= kVoxelMatTableCap[code])
			return 0;
		return table(num);
	}

	/// 填全局材质表（按方块编号，0 = 空）：摩擦 / 弹性。**所有 VoxelShape 共用一份**
	/// ⇒ 每个形状零额外内存。数据来源是 Unity 侧 VxlMaterialTable.ExportPhysics()。
	///
	/// ★ **写时复制**：内部新建一张表、填好之后原子发布，读侧（窄相位的 CombineFriction /
	///   CombineRestitution 回调，跑在多个 worker 线程上）拿到的永远是"一张完整的表"的指针。
	///   所以本函数**可以**在物理步进期间调用（进关卡 / 改材质配置），不需要先停 PhysicsSystem。
	///   旧表不释放（一张表 256 项 × 24 字节 ≈ 6 KB，换表频率是"进关卡"级别），这样上层已经拿到的
	///   `const PhysicsMaterial *` 永远不会变成野指针。
	/// ★ **内容相同则直接返回**（不分配、不换指针）。这一条是为"体素随时增删"准备的：上层很容易
	///   写成"改完方块就把整表重导一遍"，那种调用绝大多数时候内容没变；没有这条短路的话，
	///   每次编辑都会漏一张 6 KB 的旧表。顺带让 GetMaterial() 的返回地址保持稳定。
	/// ★ 注意它和**体素增删**是两条独立的链路：增删体素（UpdateSolid / UpdateSolids）不碰这张表，
	///   只有"编号 → 摩擦/弹性的映射"本身变了才需要调用本函数。
	static void				sSetMaterialTable(const float *inFriction, const float *inRestitution, int inCount);

	/// 取全局材质表里第 inId 号材质（越界会绕回 0）
	static const VoxelMaterial *sGetVoxelMaterial(uint inId);

	/// ★ 把"按体素材质取摩擦/弹性"的组合回调装进 ioSystem。
	///   Jolt 默认只用 Body 上的摩擦/弹性（一个刚体一个值），装上之后：
	///     摩擦 = sqrt(材质1.摩擦 * 材质2.摩擦)   弹性 = max(弹性1, 弹性2)   （与默认组合规则同形）
	///   非体素形状 / 没有材质数据的一方仍退回 Body 的值 ⇒ 与旧行为兼容。
	static void				sInstallMaterialCombine(PhysicsSystem &ioSystem);

	// 见 Shape::GetMaterial
	virtual const PhysicsMaterial *	GetMaterial(const SubShapeID &inSubShapeID) const override;

	// 见 Shape::GetSurfaceNormal
	virtual Vec3			GetSurfaceNormal(const SubShapeID &inSubShapeID, Vec3Arg inLocalSurfacePosition) const override;

	// 见 Shape::GetSubmergedVolume
	virtual void			GetSubmergedVolume(Mat44Arg inCenterOfMassTransform, Vec3Arg inScale, const Plane &inSurface, float &outTotalVolume, float &outSubmergedVolume, Vec3 &outCenterOfBuoyancy JPH_IF_DEBUG_RENDERER(, RVec3Arg inBaseOffset)) const override;

#ifdef JPH_DEBUG_RENDERER
	// 见 Shape::Draw
	virtual void			Draw(DebugRenderer *inRenderer, RMat44Arg inCenterOfMassTransform, Vec3Arg inScale, ColorArg inColor, bool inUseMaterialColors, bool inDrawWireframe) const override;
#endif // JPH_DEBUG_RENDERER

	// 见 Shape::CastRay
	virtual bool			CastRay(const RayCast &inRay, const SubShapeIDCreator &inSubShapeIDCreator, RayCastResult &ioHit) const override;
	virtual void			CastRay(const RayCast &inRay, const RayCastSettings &inRayCastSettings, const SubShapeIDCreator &inSubShapeIDCreator, CastRayCollector &ioCollector, const ShapeFilter &inShapeFilter = { }) const override;

	// 见 Shape::CollidePoint
	virtual void			CollidePoint(Vec3Arg inPoint, const SubShapeIDCreator &inSubShapeIDCreator, CollidePointCollector &ioCollector, const ShapeFilter &inShapeFilter = { }) const override;

	// 见 Shape::CollideSoftBodyVertices
	virtual void			CollideSoftBodyVertices(Mat44Arg inCenterOfMassTransform, Vec3Arg inScale, const CollideSoftBodyVertexIterator &inVertices, uint inNumVertices, int inCollidingShapeIndex) const override;

	// 见 Shape::GetTrianglesStart
	//
	// 体素形状没有"派生几何"这一层（没有三角形列表可迭代）：窄相位是直接按格子做的。
	// 需要可视化请看 JPH_DEBUG_RENDERER 下的 Draw()（它只画暴露出来的面）。
	virtual void			GetTrianglesStart(GetTrianglesContext &ioContext, const AABox &inBox, Vec3Arg inPositionCOM, QuatArg inRotation, Vec3Arg inScale) const override;

	// 见 Shape::GetTrianglesNext
	virtual int				GetTrianglesNext(GetTrianglesContext &ioContext, int inMaxTrianglesRequested, Float3 *outTriangleVertices, const PhysicsMaterial **outMaterials = nullptr) const override;

	// 见 Shape::GetStats
	virtual Stats			GetStats() const override
	{
		// 体素数组是调用方的，不算在形状自己的内存里
		return Stats(sizeof(*this), 0);
	}

	// 见 Shape::GetVolume
	virtual float			GetVolume() const override
	{
		EnsureDerived();
		return float(mNumSolidVoxels) * Cubed(mVoxelSize);
	}

	// 把形状函数注册进碰撞分发表
	static void				sRegister();

	/// 调试几何的四边形数量上限（防止网格很大时一次分配太多内存）
	static constexpr uint	cMaxDebugQuads = 65536;

	/// 体素 vs 其它形状窄相位的格形状开关（默认开）。
	///
	/// 开：每格用**内切球**（sGetUnitSphere，半径 0.5 * min(格缩放分量)）代替单位盒 ——
	///     球 vs 任意形状 = 1 个点接触（盒 = 4 顶点面流形），10x10 落地面 400 条 -> 100 条，
	///     查询本身也更快（球的特化路径比盒的 GJK/EPA 轻）。代价：格子的棱角不参与碰撞，
	///     贴墙/棱上会更"圆"。
	///     这条路径还会把**沿 X 连续、体素值相同**的实心格合并成一根胶囊（见 sGetUnitCapsule），
	///     所以真正的窄相位调用次数远小于实心格数量。
	/// 关：每格用单位盒（sGetUnitBox，凸半径 0），上游行为。
	///
	/// 注意：无论开关如何，本方形状三个轴的格缩放**不均匀**时都会退回逐格（球只支持均匀缩放，
	/// 胶囊更严格），因为胶囊的长度与半径必须用同一个缩放量才能对上格子间距。
	///
	/// ★ **体素 vs 体素**不受它控制：sCollideVoxelGrids 永远走"跨格枚举格对筛选 + 整网格 SAT
	///   全局轴接触几何"这条单一路径（格子的球近似语义与这里的内切球一致）。
	///   CastRay / shape cast 一律仍按盒子处理（射线没有体积，用球会有洞；这条刻意分开，互不影响）。
	static bool				sUseSphereContacts;

	/// 体素 vs 体素：是否跳过"两个格子都是内部格"的格对（默认关，实验开关）。
	///
	/// 内部格 = 6 个邻居全是实心格的实心格。表面 x 实心、表面 x 表面都不受影响，只有**两边
	/// 都是内部格**的那一对才跳过 —— 这一对的两格都深埋在自己的实体里，产生的接触是"埋没接触"。
	///
	/// ★ 风险已知："暴露面过滤"（另一套更激进的过滤）当时把深嵌时的支撑接触全杀掉了，方块直接
	///   穿落；内部格过滤比它温和，但深嵌时的推力同样可能来自这些埋没接触，所以默认关，靠单测
	///   （掉落 / 半格错位堆叠 / 深嵌卡死）验证后再决定是否打开。
	static bool				sSkipInteriorPairs;

	/// 体素 vs 体素：强制指定"遍历侧"（0 = 自动选格数少的一侧，1 = 强制遍历 shape1，2 = 强制 shape2）。
	/// 仅供单元测试做"两个方向结果一致"的对照，运行时不要设置。
	static int				sIterateSideOverride;

	/// 体素 vs 体素：接触法线（= 深度方向、接触点推出方向）取自哪里。0~4 = 见下；
	/// 每条的实测性能与更细的取舍见 VoxelShape.cpp 里 sNormalMode 定义处的说明。
	///
	/// 0 = **两个网格整体 OBB 的 SAT 轴**。所有格对共用同一根轴 ⇒ 接触方向一致，
	///     深嵌交错时不会互相抵消，行为等于"实心大盒撞实心大盒"。
	///     代价：包围盒表达不了局部表面 —— 环面落圆锥这类**曲面/斜面**接触会给出与接触面
	///     无关的方向（甜甜圈会被圆锥顶着往上弹、球放环心会自旋）。
	/// 1 = **逐对格心连线**（球语义）。direction = normalize(格心2 - 格心1)，深度 = r1+r2-|Δc|，
	///     法线跟随局部几何。★ 已知翻车：深嵌交错时上块**沉 0.11 m ≈ 1.1 格**（不卡死、不倾倒，
	///     只是静止高度错，所以场景里不容易看出来）；错位半格的对齐堆叠被斜法线的横向分量
	///     横推（低 1.3 cm）。**四个模式里最便宜**（复用了深度本来就要算的 Δc）。
	/// 2 = **暴露面法线**。取 shape1 那一格"朝向 shape2、且邻居为空"的格面，法线落在格子局部轴上；
	///     没有任何朝向对方的暴露面（整格埋住）时退回全局轴。平面接触给出精确格面法线，斜/曲面
	///     接触给出阶梯状局部法线（相邻接触不再全部同向 ⇒ 聚合后等效于局部表面）。六个实测场景
	///     方向全部正确；代价是比 mode 1 贵 9~12%（每轴 <= 3 次 IsSolid + 3 次 dot）。
	/// 3 = **融合模式（"融化"）**：按这一对格子的**外露面数量**在 mode 1 / mode 0 之间分流 ——
	///     **任一侧**外露 >= 2 面（棱格 / 角格，= 物体的边缘）用 mode 1 的逐对格心连线（方向可以是
	///     任意角度 ⇒ 边缘保持圆润），所以"平面格 x 边缘格"这一对也是球语义，不管遍历侧落在哪一边；
	///     **两侧都**外露 < 2 面（平面表面格 + 整格埋住的深嵌格）**直接用整网格 SAT 全局轴
	///     （mode 0）**：整格埋住时 mode 2 本来就退回全局轴（逐位相同），平面表面格是平坦接触、
	///     全局轴就对 ⇒ 省掉 3 次 dot。取两者之长：既不像纯 mode 1 那样把对齐平面
	///     算成 45° 斜推、深嵌时把方块沉下去，也不像纯 mode 2 那样把曲面边缘量化到 6 个格轴。
	///     ★ 判据只需"外露是否 >= 2"⇒ **数到 2 就停**（棱 / 角格平均 2~4 次 IsSolid）；前 5 个邻居
	///     全实心时第 6 个不必查（即使是空也只是"外露 1"，仍非边缘）。
	///     ★ 成本：**与 mode 2 基本持平**（0.93~1.00）。遍历侧那一格按格懒算一次（相邻候选复用，
	///     一个遍历格最多 8 个候选）；只有它是非边缘格时才再数**对方**那一格（逐对，靠
	///     sEdgeCacheEnabled 摊薄）—— 这一项就是比"只判遍历侧"贵出来的 3~9%（见 sEdgeEitherSide）。
	///     ★ 已知弱点：一格厚的板每格天生上下都外露 = 2 ⇒ 只要碰到它就整体走 mode 1，实测沉
	///     1.35 cm（见 UnitTests 的 `VoxelShapeNormalModeSimulation` 的 "thin iterated" 场景）。
	/// ★ 无论取哪个，**格对的集合完全不变**（跨格枚举 + 球心距筛选与法线无关）⇒ 0/1/2/3 是同一次
	///   遍历下的严格对照（实测接触数 210/210/210/210、1024 各四份），接触数差异只来自深度筛选。
	/// ★ 性能（count-only 多模式交替、每轮轮换顺序、min-of-9，`VoxelShapeEdgeRuleCost`，相对 mode 2）：
	///     10^3（210 接触）：mode 0 0.91 | mode 1 0.94 | mode 2 1.00 | mode 3（判两侧）0.98
	///     20^3（1024 接触）：mode 0 0.87 | mode 1 0.89 | mode 2 1.00 | mode 3（判两侧）0.95
	///   （用 min 而不是中位数：环境噪声会让整轮漂移 1.5x，中位数会把漂移带进比值。）
	/// ★ mode 3 独有的"数外露面"这一步有缓存，见 sEdgeCacheEnabled（命中时 0 次邻居查询）。
	/// 4 = **局部 2x2x2 邻域分类**（**默认**）。把遍历侧那一格（b）映射到候选侧格空间 ——
	///     在那里 b 是轴对齐的 AABB（半宽 = 0.5 * (|step0| + |step1| + |step2|)）—— 用它圈出候选侧
	///     的 2x2x2 邻域（8 格，每轴跨 1~2 格，只跨 1 格时按格心偏向补一格），数这 8 格里**候选侧**
	///     的实心格，按排布定法线（全部在格空间里算，不需要投影）：
	///       1 格 -> 球语义（b 格心离开该格心）；2 格同轴 -> 胶囊（离开线段最近点）；
	///       4 格同面 -> 面法线（指向空格那一侧）；7 格 -> 推向唯一的空格；
	///       6 格（两块空格）-> 把两空格连成线段、推向最近点；8 格 -> 完全埋住，用整网格全局轴；
	///       其余（3 / 5 格、对角 2 格、两条平行棱…）-> 贪心拆成"面 > 棱 > 单格"的基本块，
	///       每块算一个方向、按块内格数加权合并成一条。
	///     ★ 动机：整网格 SAT 的 half 由 size 决定、**不含空气**，大网格里"90% 是空气"时全局轴退化
	///     成"离哪个盒面最近"，与真实接触面无关（8 m 地形块 vs 0.5 m 碎块 ⇒ 碎块贴边时法线变纯水平、
	///     左右横跳）。2x2x2 只看得见眼前的 8 格 ⇒ 方向由真实重叠轮廓决定。
	///     ★ 仍遵守"两网格整体 SAT 先定一根轴"的契约：只有"完全埋住（8 格）"才用它。
	///     ★ 性能（`VoxelShapeNormalModePerformance`，Release，us/query，min-of-5x200x3）：
	///     地形 x 碎块 64 对：平地表 15.90 -> 16.32（+2.6%）、坑壁 15.86 -> 17.25（+8.8%）；
	///     体素 x 体素：1024 对深重叠 31.37 -> 43.69（+39%）、64 对浅接触 3.41 -> 5.49（+61%，绝对 +2.1 us）。
	///     差距来源 = **每格对 8 次位图查询**（已按 4x4x4 子块做字缓存，平均 1.95 次 load）
	///     + 1~5 格分支的贪心拆块。C 场景探针（跳过 8 格查询）实测 28.4 us ⇒ 比 mode 3 还快。
	static int				sNormalMode;
	/// mode 1 的深穿透回退阈值（单位 = 格长；0 = 关闭回退）。逐对格心连线只描述"浅接触"，
	/// 重叠超过这个深度就退回整网格 SAT 轴，避免深层格对的法线互相抵消（合力比 ≈ 0）。
	/// 0.25（默认）= 四分之一格深。实测三分界：
	///   0.50 → "双向半格错位（楔住）"这种对称互嵌的合力比仍是 0.08，漏掉；
	///   0.15 → 旋转 L 块静止贴合的格对也会触发，抖动从 0.00 涨到 4.37 mm（退回 mode 0 行为）；
	///   0.25 → 互嵌合力比 1.00，静止抖动仍是 0.00~0.03 mm。
	static float			sDeepOverlapFraction;

	/// mode 4 的 2x2x2 块掩码缓存（A/B 用，默认 1 = 开）。语义完全相同，只影响性能。
	///
	/// 背景：mode 4 的块起点 lo4[k] = a[k] 或 a[k]-1（a = 候选侧这一格），而枚举窗口的起点是
	/// floor(c-0.5)（c = 遍历格心在候选格空间的坐标）—— 两者对真实重叠的格对**逐位相同**，也就是
	/// "同一个遍历格的所有候选用的是同一个块"。所以在一个遍历格内记下上一次的 lo4，命中就跳过
	/// build_block_mask 的 8 次位查询（字缓存后约 2 次 load）。
	/// 命中率取决于"每个遍历格有几个候选"：1 个候选 ⇒ 0%，2 个 ⇒ 50%。
	static int				sBlockMaskCache;

	/// 融合模式的路由判据看**两侧**（1，默认）还是只看**遍历侧那一格**（0）。仅供 A/B 与回溯定位用。
	///
	/// 1 = 任一侧外露 >= 2 面就走球语义（mode 1）⇒ "平面格 x 边缘格"这一对按球语义算，边缘（棱 /
	///     角 / 斜台阶）在**任何一侧**都保持圆润，结果不再依赖"这次遍历选了哪一边"。
	///     代价：遍历侧是平面格时要多看对方那一格（逐对，靠缓存摊薄）。实测**比判一侧贵 3~9%**
	///     （`VoxelShapeEdgeRuleCost`：both / owner-only = 1.03~1.09，10^3 与 20^3、10° 与 45° 都量了），
	///     换算成相对 mode 2 就是 0.95~0.98 —— 判一侧时是 0.88~0.94。
	/// 0 = 旧判据，只看遍历侧那一格：遍历侧是平面格时整对退回全局轴（"平面格 x 边缘格"被当成
	///     实心盒的平面接触）。开销更低，但结果依赖遍历侧怎么选。
	static int				sEdgeEitherSide;

	/// ---- 边缘格判定缓存：只被 sNormalMode == 3 用到（其余模式不数外露面）----
	///
	/// 融合模式判据是"这一格有几个面外露"，而这个量**只取决于体素数据**（形状局部格坐标）——跟形状
	/// 在世界里的位姿、缩放、跟谁碰撞都无关 ⇒ 同一个格子在每一帧、每一对物体的查询里答案都一样。
	/// 所以把它记下来：命中就直接拿答案，省掉 2~6 次 IsSolid（邻居查询）。
	///
	/// 为什么不按格预存：那要给**整张网格**每格留位（1 bit/格，大网格上就是几十 MB），而且大部分
	/// 区域永远不会参与碰撞。这里换成**惰性**的：一个进程级的直接映射缓存（2^20 槽 x 8 字节 = 8 MB），
	/// **首次用到才分配**，只保存真正被查询过的格子。
	///
	/// 键 = (体素数据指针, 该形状的体素数据版本, x, y, z) 混成的 48 位哈希；版本在每次 Recalculate()
	/// 时换一个新的全局唯一值 ⇒ 数据改过以后旧条目自动作废（哈希变了，对不上），**不需要清表**。
	/// 哈希**对 x 是仿射的**（x 的系数为 1），相邻的 x 落在相邻的槽里 ⇒ 连续格子共享缓存行，
	/// 这是命中便宜的关键。槽是原子的 64 位字（窄相位是多线程的），直接映射、**不探测**：
	/// 冲突就覆盖 —— 本来就是缓存，丢了再算一次而已。
	///
	/// 收益完全取决于**同一个格子是否被反复查询**：持续接触的物体每帧都在问同一批格子 ⇒ 稳态接近
	/// 全命中，只付一次 load + 比较；一次性查询（新出现的接触对、射线）是全 miss，比不开缓存多付
	/// 一次插入。实测见 UnitTests 的 VoxelShapeEdgeCacheCost / VoxelShapeEdgeCacheEquivalence：
	///   * 命中率 100% 时 10^3 快 2~5%、20^3 快 1%（cached / raw；8 MB 表和 512 KB 表都一样，
	///     交替运行两个 exe 量过 —— 表的**足迹**不是瓶颈，查找本身和那几次邻居查询同量级）；
	///   * 冷缓存（每个查询前 Recalculate）慢 1~3%（插入的代价）；
	///   * ★ 整个"数外露面"步骤本来只占查询的 2~6%（见 sNormalMode 的性能表：mode 0 与 mode 3
	///     的差），所以缓存能省的上限就只有这么多 —— 别指望它是大优化。
	///   * 效果在内存延迟高的时候变明显（同一台机器上机器负载重时量到过 5%，空载 1~2%）：
	///     数外露面是一条"取邻居 -> 分支 -> 再取"的依赖链，缓存把它换成一次 load。
	///
	/// 0 = 关（每次都直接数邻居，与没有这个缓存时逐位相同）；1 = 开（默认）。
	static int				sEdgeCacheEnabled;

	/// 非 0 时累计下面的命中 / 未命中计数。**只在单线程的测试里打开**：计数器是原子的（relaxed），
	/// 查询路径上为它多一个可预测的分支；默认关闭，时序测量不受影响。
	static int				sEdgeCacheStats;
	static atomic<uint64>	sEdgeCacheHits;						///< 命中次数（sEdgeCacheStats != 0 时累计）
	static atomic<uint64>	sEdgeCacheMisses;					///< 未命中次数（含"算完插入"的那些）

	/// 诊断用：上一次 sCollideVoxelGrids 里整网格 SAT 的 [0] = 最小重叠（选中的轴）、
	/// [1] = 次小重叠。比值 best/second 是"扁平（面/斜面）接触"还是"三轴深嵌"的候选判据。
	/// 只在 sNormalMode != 0 之外的诊断场景读；正常运行时不要依赖。
	/// ★ 窄相位是多线程的 ⇒ 这里必须是原子的：普通 float 数组被多个 worker 同时写是数据竞争
	///   （同一个对象被并发写），TSAN/并发分析会直接报出来。
	static atomic<float>	sLastGridOverlap[2];

private:
	// ---- 分块几何常数：8x8x8 的 chunk 和 4x4x4 的子块 ----
	//
	// 占用位图里**一个 uint64 就是一个 4x4x4 子块**（64 格填满 64 bit），所以 4x4x4 同时是
	// "存储的原子单位"和"跳过的原子单位"：
	//   跳过一层：一个 4x4x4 子块为空 -> 一次 64 位比较，跳过 64 格；
	//   跳过一层：字里的一个 nibble（一行 4 格）为空 -> 一次移位 + 掩码，跳过 4 格。
	// ★ 早先那版这里还有第二份索引 mChunkMasks（每个 chunk 8 个连续 uint64 的占用掩码），
	//   它**已删除** —— 那份掩码是对体素数据的派生副本，就地编辑后可能过期（往掩码为 0 的 chunk 里
	//   加一格会被整块跳过 ⇒ 漏碰撞）。现在两级跳过直接读占用位图本身，不存在过期问题。
	// ★ 8x8x8 这个粒度**不占物理侧任何存储**，也**不参与物理**：一个 chunk = 2x2x2 个 texel，它只是
	//   渲染侧材质图的粒度（每个 chunk 一个方块编号）。物理材质是整形状一个（见 GetMaterial），
	//   逐格/逐 chunk 的方块类型从来不参与摩擦/弹性。
	static constexpr int	cChunkShift = 3;										///< 每个 chunk 每个轴 8 格
	static constexpr int	cChunkSize = 1 << cChunkShift;							///< 8
	static constexpr int	cChunkMask = cChunkSize - 1;							///< 7
	static constexpr uint	cChunkVolume = cChunkSize * cChunkSize * cChunkSize;	///< 512
	static constexpr int	cSubChunkShift = 2;										///< 子块每个轴 4 格
	static constexpr int	cSubChunkSize = 1 << cSubChunkShift;					///< 4
	static constexpr int	cSubChunkMask = cSubChunkSize - 1;						///< 3
	static constexpr int	cNumSubChunksPerAxis = cChunkSize / cSubChunkSize;		///< 每轴 2 个子块
	static constexpr uint	cSubChunkVolume = cSubChunkSize * cSubChunkSize * cSubChunkSize;	///< 64 = 一个 uint64 的位数
	static constexpr int	cChunkWords = cChunkVolume / 64;						///< 一个 chunk 8 个字（512 bit = 2x2x2 个 texel）

	/// 占用位图沿一个轴的 texel 数 = ceil(格数 / 4)（至少 1，免得退化尺寸下算出 0）
	static inline int		sGetTexelCount(int inSize)
	{
		return JPH::max(1, (inSize + cSubChunkMask) >> cSubChunkShift);
	}

	/// 子块在 chunk 里的序号（0..7）
	static inline uint		sGetSubChunkIndex(int inX, int inY, int inZ)
	{
		return uint(((inX >> cSubChunkShift) & (cNumSubChunksPerAxis - 1))
			| (((inY >> cSubChunkShift) & (cNumSubChunksPerAxis - 1)) << 1)
			| (((inZ >> cSubChunkShift) & (cNumSubChunksPerAxis - 1)) << 2));
	}

	/// 格在所属子块的那个 uint64 里的位序号（0..63）：lx + ly * 4 + lz * 16
	/// ★ x 占最低 2 位 ⇒ 一行连续 4 个 x 正好是一个 nibble，可以一次判空整行（见 sVisitSolidVoxelRunsInBox）。
	static inline uint		sGetSubChunkBit(int inX, int inY, int inZ)
	{
		return uint((inX & cSubChunkMask) | ((inY & cSubChunkMask) << cSubChunkShift) | ((inZ & cSubChunkMask) << (2 * cSubChunkShift)));
	}

	/// 返回编码 inSize 个格子位置所需要的最小二进制位数（例如 10 个格子需要 4 位）
	static uint				sGetBitsForSize(int inSize);

	/// 三轴格数（按下标取）
	inline int				GetAxisSize(int inAxis) const			{ return inAxis == 0? mSizeX : inAxis == 1? mSizeY : mSizeZ; }

	/// 惰性统计：实心格数 / 实心包围范围 / 质量属性。只在真正需要时扫一遍（按占用位图的 4x4x4 字扫），
	/// 之后一直复用，直到 Recalculate() 让它失效或就地改体素做增量维护。
	/// ★ 静态地形区块不会调用 GetMassProperties / GetNumSolidVoxels，所以整块**一次都不扫**。
	void					EnsureDerived() const;

	/// UpdateSolid() / UpdateSolids() 的公共实现：就地改一格并增量维护缓存，**不含**数据版本切换。
	/// 返回 true 表示这一格的**实心性**发生了变化（调用方据此决定是否需要换数据版本）。
	/// 越界坐标会被忽略并返回 false；写入值与当前值相同时不做任何缓存更新。
	bool					sApplyVoxelEdit(int inX, int inY, int inZ, bool inSolid) const;

	/// 手写的"体素网格 vs 体素网格"窄相位。
	///
	/// 这是 VoxelShape 性能和行为的关键路径，刻意**不走**碰撞分发表：
	/// 1. 先对两个网格的整体 OBB 做一次 SAT，求出唯一的最小分离轴（世界空间）—— 所有接触点共用这一根轴。
	///    ★ 这一步不能省：如果让每个格子各自算最小分离轴，深穿透（两个立方体大面积重合）时三轴会打平，
	///      每格挑到的轴都不一样，法线互相矛盾，求解器收到的净推进力抵消 ⇒ 表现为"两个方块叠在一起
	///      互相插着、推不开"。
/// 2. 再遍历两个网格重叠区里的实心格对（用分块掩码跳过空块）。**遍历侧**取重叠区格数少的那一侧
///    （大网格 vs 小方块时省一半以上的遍历）。格对用**跨格枚举**筛选：把遍历侧的格心映射到候选侧
///    的格坐标系，每轴取 floor(c-0.5-e) .. ceil(c+0.5+e)-1（e = 全局轴方向上的推测余量，对齐时
///    水平方向 e = 0），再在格下标空间做一次球心距回收（跨格集合是盒子重叠集，比内切球宽）。
///    每个候选产出 1 个点接触：法线/深度沿全局轴（与实心大盒同一根轴、同一个公式），接触点 =
///    格心沿轴推到各自的格面；不填 mShape1Face/mShape2Face（面为空时 Jolt 退回单点接触）。
///    遍历顺序**由外往中心**：接触数被 cMaxVoxelContacts 截断时保留的仍是整片外围，不会只剩单边。
	static void				sCollideVoxelGrids(const VoxelShape *inVoxel1, Vec3Arg inScale1, Mat44Arg inCenterOfMassTransform1, const SubShapeIDCreator &inSubShapeIDCreator1,
											   const VoxelShape *inVoxel2, Vec3Arg inScale2, Mat44Arg inCenterOfMassTransform2, const SubShapeIDCreator &inSubShapeIDCreator2,
											   const CollideShapeSettings &inCollideShapeSettings, CollideShapeCollector &ioCollector);

	/// 所有实心体素共享的单位盒子：边长 1、中心在原点、**凸半径 0**。
	/// 凸半径必须是 0，否则每格都会被撑大 cDefaultConvexRadius（默认 0.05 m），体素之间会互相"膨出"。
	static const BoxShape *	sGetUnitBox();

	/// sUseSphereContacts 开启时，体素 vs 其它形状窄相位用的单位球：半径 0.5（内切球）、中心在原点，
	/// 每格乘上 0.5 * min(格缩放分量)。球给出 1 个点接触（盒给 1 个 4 点面流形），接触数和查询成本都更低；
	/// 代价是格子的棱角不参与碰撞。cast 路径不用它（射线没有体积，见 sGetUnitBox 的使用处）。
	static const SphereShape *	sGetUnitSphere();

	/// 游程合并用的胶囊池：长度 2..cMaxRunLength 各一个，半径 0.5、圆柱半高 (L - 1) / 2（单位 = 格），
	/// 乘上**均匀**的格缩放之后正好覆盖 L 个连续的格子（球心落在格心上、半径 = 半格，圆段之间被圆柱
	/// 填满）。一次建成、只读访问 —— 窄相位是多线程的，所以这里不能有惰性写入。
	/// 游程合并的收益来源：形状分发 / mesh 树遍历 / 结果构造这些**每次调用**的固定开销，逐格付一遍
	/// 变成每 L 格付一遍（见 UnitTests 的 VoxelShapeMeshRunPrimitiveCost 实测）。
	static const CapsuleShape *	sGetUnitCapsule(int inRunLength);

	/// 一根胶囊最多覆盖多少格。更长的游程会拆成多段（尾段可能只剩 1 格，那时用内切球）。
	static constexpr int	cMaxRunLength = 64;

	/// 遍历 inLocalBox 覆盖到的所有**实心**体素，对每格调用一趟 inVisitor：
	///
	///		inVisitor(x, y, z, inCellTransform, inCellScale)
	///
	/// - inLocalBox：体素形状的**未缩放局部空间**（也就是 GetVoxelCenter 返回坐标的那个空间，单位：米）里的
	///   轴对齐盒。窄相位用它把遍历范围限制在"两个形状包围盒的重叠区"里。
	/// - inVoxelScale / inVoxelTransform：施加在体素形状上的缩放与质心变换，用来算格子的目标空间变换。
	/// - inCellTransform：该格共享的"单位盒子"（边长 1、中心在原点）到目标空间的**旋转 + 平移**部分
	///   （平移量就是格心）。注意这里**不含**边长缩放，否则每格都要多存一个矩阵。
	/// - inCellScale：施加在该格单位盒子上的缩放，等于 inVoxelScale * mVoxelSize。
	///   ★ 单位盒子乘上它才等于真正的体素格子大小 —— 所以取格子的面/顶点时**必须**把这个缩放带上，
	///     只用 inCellTransform 会得到一个边长 1 米的盒子（接触面变大、接触点散到网格外面去）。
	///
	/// 这套"把格子的变换算好再交给访问者"的设计，是为了让所有窄相位回调都只关心"拿到这一格怎么用"，
	/// 不必各自重复一遍格心/缩放/SubShapeID 的换算。
	///
	/// 遍历按 8x8x8 的 chunk 推进，块内再按 4x4x4 的子块跳过空子块（每格所属的那个 64 位字直接
	/// 从占用位图里取，见 GetWord），所以在一张大而稀疏的网格上，开销只取决于
	/// "重叠区里真正有实心格的那几个子块"。
	///
	/// tOutsideIn = true 时从重叠区**由外往中心**遍历（每轴取 下界, 上界, 下界+1, 上界-1, ...），
	/// 而不是的下标顺序遍历。用途：接触数被 cMaxVoxelContacts 截断时，先被保留的是**外圈**的格子
	/// —— 截断后剩下的接触仍然分布在整片重叠区的外围（形心与整片重合），不会像顺序截断那样只剩
	/// 一侧的接触（那会让物体被支撑在单边、发生倾斜/下陷）。默认 false = 保持上游顺序。
	template <bool tOutsideIn = false, class F>
	void					sVisitSolidVoxelsInBox(const AABox &inLocalBox, Vec3Arg inVoxelScale, Mat44Arg inVoxelTransform, const F &inVisitor) const;

	/// 与 sVisitSolidVoxelsInBox 同一个遍历范围，但把**沿局部 X 轴连续、且体素值相同**的实心格合并成
	/// "游程"，每个游程只调用一次 inVisitor：
	///
	///		inVisitor(inX, inRunLength, inY, inZ, inFirstCellTransform, inRunCenter, inRunAxis)
	///
	/// - inX / inY / inZ：游程**第一格**的下标（也是 SubShapeID 编码用的坐标）。游程内体素值相同，
	///   所以用首格的 SubShapeID 取材质/法线都是对的。
	/// - inRunLength：游程长度（≥ 1）。超过 cMaxRunLength 的游程会被拆成多段。
	/// - inFirstCellTransform：第一格的变换（单位盒 -> 目标空间的旋转 + 平移）。只有 inRunLength == 1
	///   （退化回单格内切球）时会用到；多格时用 inRunCenter / inRunAxis 构造胶囊。
	/// - inRunCenter：游程中心的目标空间坐标（= 首尾格心的中点，也是胶囊中心）
	/// - inRunAxis：目标空间里"x 下标 +1"的方向（单位向量；胶囊局部 Y 轴对齐到它）。整个查询里是常量。
	///
	/// 为什么沿 X：占用位图里 x 占**最低**的 2 位（一行连续 4 个 x 正好是一个 nibble），所以扫一行
	/// 只需要"每个 4 格组取一个字 + 移位"，比逐格判实心省掉 3/4 的访存。
	/// ★ 游程是"连续实心格"的最大段（**不再按体素值断段** —— 材质已经移到按 chunk 的材质图上，
	///   体素里没有逐格的值了）。
	template <class F>
	void					sVisitSolidVoxelRunsInBox(const AABox &inLocalBox, Vec3Arg inVoxelScale, Mat44Arg inVoxelTransform, const F &inVisitor) const;

	/// 求体素未缩放局部空间里的一个轴对齐盒所覆盖的格子下标区间（闭区间，已经 clamp 进网格，并且和实心格范围求交）。
	/// 返回 false 表示没有交集，可以整个跳过。
	bool					sGetCellRange(const AABox &inLocalBox, int &outMinX, int &outMinY, int &outMinZ, int &outMaxX, int &outMaxY, int &outMaxZ) const;

	/// 找到 inLocalPosition 所在格子里"最靠近该点、且是暴露面（邻居为空）"的那个面，返回它的法线。
	/// 返回 false 表示这个格子六个邻居全是实心（点在实体内部深处，没有可以直接推出去的面）。
	bool					FindClosestVoxelFace(int inX, int inY, int inZ, Vec3Arg inLocalPosition, Vec3 &outNormal) const;

	/// 判断一个格子的某个面是否是暴露面（该方向的邻居为空）
	inline bool				IsVoxelFaceExposed(int inX, int inY, int inZ, int inDX, int inDY, int inDZ) const
	{
		return !IsSolid(inX + inDX, inY + inDY, inZ + inDZ);
	}

	/// 判断一个实心格是否是**内部格**：6 个邻居全是实心（埋在实体内部，没有任何暴露面）。
	/// 只在 sSkipInteriorPairs 打开时参与体素 vs 体素的格对过滤（见该开关的说明）。
	inline bool				IsVoxelInterior(int inX, int inY, int inZ) const
	{
		return IsSolid(inX - 1, inY, inZ) && IsSolid(inX + 1, inY, inZ)
			&& IsSolid(inX, inY - 1, inZ) && IsSolid(inX, inY + 1, inZ)
			&& IsSolid(inX, inY, inZ - 1) && IsSolid(inX, inY, inZ + 1);
	}

	// ---- 窄相位分派函数（在 sRegister() 里注册进 CollisionDispatch 的分发表）----
	//
	// 体素形状不做"整体一次凸查询"，而是遍历重叠区里的每一格：
	// - 对方也是体素时 -> 走下面的手写快路径 sCollideVoxelGrids（不查表、不造形状对象）
	// - 对方是凸/网格形状时 -> 每格与共享的单位盒子一起交给常规凸通道（见 .cpp 里的详细说明）
	// 三个函数的形状参数顺序与遍历的对象不同，注释在 .cpp 里有详细说明。

	/// (任意形状, 体素) 的 collide：遍历**体素**（参数 inShape2）的实心格
	static void				sCollideConvexVsVoxel(const Shape *inShape1, const Shape *inShape2, Vec3Arg inScale1, Vec3Arg inScale2, Mat44Arg inCenterOfMassTransform1, Mat44Arg inCenterOfMassTransform2, const SubShapeIDCreator &inSubShapeIDCreator1, const SubShapeIDCreator &inSubShapeIDCreator2, const CollideShapeSettings &inCollideShapeSettings, CollideShapeCollector &ioCollector, const ShapeFilter &inShapeFilter);

	/// (体素, 任意形状) 的 collide：遍历**体素**（参数 inShape1）的实心格
	static void				sCollideVoxelVsVoxel(const Shape *inShape1, const Shape *inShape2, Vec3Arg inScale1, Vec3Arg inScale2, Mat44Arg inCenterOfMassTransform1, Mat44Arg inCenterOfMassTransform2, const SubShapeIDCreator &inSubShapeIDCreator1, const SubShapeIDCreator &inSubShapeIDCreator2, const CollideShapeSettings &inCollideShapeSettings, CollideShapeCollector &ioCollector, const ShapeFilter &inShapeFilter);

	/// (任意形状, 体素) 的 cast：遍历**体素**（参数 inShape）的实心格
	static void				sCastConvexVsVoxel(const ShapeCast &inShapeCast, const ShapeCastSettings &inShapeCastSettings, const Shape *inShape, Vec3Arg inScale, const ShapeFilter &inShapeFilter, Mat44Arg inCenterOfMassTransform2, const SubShapeIDCreator &inSubShapeIDCreator1, const SubShapeIDCreator &inSubShapeIDCreator2, CastShapeCollector &ioCollector);

#ifdef JPH_DEBUG_RENDERER
	/// 把"暴露出来的面"输出成三角形列表（在体素形状的局部空间里，已经含 voxelSize 缩放）。
	/// 只输出暴露面：内部面永远看不到；顶点按"从外面看逆时针"（CCW）排列，否则会被背面剔除掉。
	void					BuildDebugGeometry(Array<DebugRenderer::Triangle> &outTriangles) const;
#endif // JPH_DEBUG_RENDERER

	// ---- 由构造参数决定、之后不再改变的数据 ----
	int						mSizeX = 0;						///< 沿 X 轴的体素数量
	int						mSizeY = 0;						///< 沿 Y 轴的体素数量
	int						mSizeZ = 0;						///< 沿 Z 轴的体素数量
	float					mVoxelSize = 0.0f;				///< 单个体素的边长（米）
	const uint64 *			mOccupancy = nullptr;			///< 占用位图，1 bit/体素（不拥有所有权）
	Array<uint32>			mMaterialHeader;				///< 逐体素材质：每 8³ chunk 一个 uint32（**形状自己的拷贝**，见 SetMaterialData）
	Array<uint64>			mMaterialData;					///< 逐体素材质：记录池（索引表 + 编号位流），同上
	int						mTexelsX = 1;					///< 占用位图沿 X 轴的 texel 数 = ceil(mSizeX / 4)
	int						mTexelsY = 1;					///< 占用位图沿 Y 轴的 texel 数 = ceil(mSizeY / 4)
	int						mTexelPlane = 1;				///< 一个 z 层的 texel 数 = mTexelsX * mTexelsY（省一次乘法）
	float					mDensity = 1000.0f;				///< 密度（kg/m^3）
	RefConst<PhysicsMaterial> mMaterial;					///< 碰撞材质

	// ---- Recalculate() 计算出来的缓存（因为 Recalculate() 是 const，所以这里是 mutable）----
	mutable uint			mVersion = 0;					///< 每次 Recalculate() 取一个新的全局唯一值：让调试几何失效，也让边缘格缓存里的旧条目失效
	mutable uint			mNumSolidVoxels = 0;			///< 实心体素数量
	mutable int				mSolidMin[3] = { 0, 0, 0 };		///< 实心体素包围范围的最小格子下标（X, Y, Z）
	mutable int				mSolidMax[3] = { -1, -1, -1 };	///< 实心体素包围范围的最大格子下标（闭区间；没有实心格时 max < min）
	mutable Vec3			mHalfExtent = Vec3::sZero();	///< 0.5 * size * voxelSize，形状局部空间里的半长
	mutable uint			mBitsX = 0;						///< 编码 X 坐标需要的 SubShapeID 位数
	mutable uint			mBitsY = 0;						///< 编码 Y 坐标需要的 SubShapeID 位数
	mutable uint			mBitsZ = 0;						///< 编码 Z 坐标需要的 SubShapeID 位数
	mutable MassProperties	mMassProperties;				///< 质量与转动惯量（转动惯量绕形状原点，即网格中心）
	mutable Vec3			mCenterOfMass = Vec3::sZero();	///< 物理质心偏移（相对网格中心），默认 0 = 网格中心，见 SetCenterOfMass
	mutable double			mInertiaDiag[3] = { 0.0, 0.0, 0.0 };	///< 惯量对角元的累加值（用 double：增量编辑一格一格加减，float 在大量删除时会有可观的消减误差）
	mutable double			mMassCenterSum[3] = { 0.0, 0.0, 0.0 };	///< Σ 格心坐标，三个轴各一个（GetMassCenter 的分子；与惯量同一套 double + 增量维护）
	mutable bool			mDerivedValid = false;					///< 实心格数 / 包围范围 / 质量属性 / 质心累加是否已统计（惰性，见 EnsureDerived）
	mutable bool				mBorderContinues = false;				///< 本网格是连续地形的一部分（±X / ±Z 那边还有隔壁区块），见 SetBorderContinues

#ifdef JPH_DEBUG_RENDERER
	mutable DebugRenderer::GeometryRef	mGeometry;			///< 缓存的调试几何
	mutable uint			mGeometryVersion = uint(-1);	///< mGeometry 对应的 mVersion
#endif // JPH_DEBUG_RENDERER
};

JPH_NAMESPACE_END
