// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/Collision/Shape/SubShapeID.h>
#include <Jolt/Physics/Collision/PhysicsMaterial.h>
#include <Jolt/Core/Array.h>

#ifdef JPH_DEBUG_RENDERER
	#include <Jolt/Renderer/DebugRenderer.h>
#endif // JPH_DEBUG_RENDERER

JPH_NAMESPACE_BEGIN

class BoxShape;
class CollideShapeSettings;
class SphereShape;

/// 构造 VoxelShape 的设置对象（"未烹调"的体素形状描述）。
///
/// 体素形状是一个规则的立方体素网格。形状**不拷贝**也**不拥有**体素数据：在形状存活期间，调用方必须保证这块
/// 缓冲区一直有效、地址不变、长度不变。体素字节 0 表示空，非 0 表示实心（上层可以把这个字节当作材质或调色板
/// 索引来用）。
///
/// 索引布局：voxels[x + y * sizeX + z * sizeX * sizeY]。
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

	/// 构造一个 sizeX * sizeY * sizeZ 个体素、每格 voxelSize 米、体素数据取自 inVoxels 的设置
							VoxelShapeSettings(int inSizeX, int inSizeY, int inSizeZ, float inVoxelSize, const uint8 *inVoxels, float inDensity = 1000.0f) :
		mSizeX(inSizeX),
		mSizeY(inSizeY),
		mSizeZ(inSizeZ),
		mVoxelSize(inVoxelSize),
		mVoxels(inVoxels),
		mDensity(inDensity)
	{
	}

	/// 见 ShapeSettings::Create
	virtual ShapeResult		Create() const override;

	int						mSizeX = 0;						///< 沿 X 轴的体素数量
	int						mSizeY = 0;						///< 沿 Y 轴的体素数量
	int						mSizeZ = 0;						///< 沿 Z 轴的体素数量
	float					mVoxelSize = 0.0f;				///< 单个体素的边长（米）
	const uint8 *			mVoxels = nullptr;				///< 体素数据，0 = 空，非 0 = 实心（不拥有所有权）
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
/// 1. **绝不拷贝体素数据**。形状只持有调用方那张数组的裸指针（mVoxels），不复制、不拥有。
///    Unity 侧渲染和物理共用同一份数据；一旦复制，内存直接翻倍 —— 对大世界来说这是不可接受的。
///    Recalculate() 算出来的只是一小撮统计量和索引，不是数据副本。
/// 2. **绝不为单个体素创建形状对象**。整个类里没有任何一处 `new BoxShape` 是按体素来的。
///    - 体素 vs 体素走的是一条**手写的窄相位**（sCollideVoxelGrids）：直接比较两个格子盒的重叠，
///      不查碰撞分发表、不构造任何形状对象、不做 GJK/EPA。
///    - 其他形状（Sphere / Box / Capsule / Mesh / ...）撞上来时，所有格子共用**同一个**进程级单位盒子
///      （sGetUnitBox()，边长 1、中心在原点、凸半径 0）或进程级单位球（sGetUnitSphere()，sUseSphereContacts
///      开启时，半径 0.5 的内切球），每格只是"缩放 + 平移"一下，没有任何一格拥有自己的形状对象。
/// 3. **只算重叠部分的体素**。窄相位只遍历"两个形状包围盒重叠区"里的实心格（sVisitSolidVoxelsInBox），
///    再用 8x8x8 的分块占用掩码（mChunkMasks）整块跳过空区域。所以与一面墙碰撞的开销和地图多大无关，
///    只与重叠区里的实心格数量有关（见 UnitTests 里的 VoxelShapeQueryScaling）。
///
/// 另外两个接口约定：
///
/// - **SubShapeID 编码体素坐标**（见 EncodeSubShapeID / DecodeSubShapeID），破坏系统据此定位被击中的格子。
/// - **质量属性是缓存值**。碰撞查询读的是实时体素数据，但质量/惯量只在 Recalculate() 时才重算，所以
///   就地删体素后刚体的行为会立刻变化，而质量要等下一次 Recalculate() 才更新。
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

	/// 便捷构造函数：直接给出网格尺寸、体素边长和体素数据
							VoxelShape(int inSizeX, int inSizeY, int inSizeZ, float inVoxelSize, const uint8 *inVoxels, float inDensity = 1000.0f);

	/// 沿 X / Y / Z 轴的体素数量
	int						GetSizeX() const						{ return mSizeX; }
	int						GetSizeY() const						{ return mSizeY; }
	int						GetSizeZ() const						{ return mSizeZ; }

	/// 单个体素的边长（米）
	float					GetVoxelSize() const					{ return mVoxelSize; }

	/// 体素数据（不拥有所有权），0 = 空，非 0 = 实心
	const uint8 *			GetVoxels() const						{ return mVoxels; }

	/// 体素在数组里的下标：x + y * sizeX + z * sizeX * sizeY
	inline int				GetVoxelIndex(int inX, int inY, int inZ) const
	{
		return inX + inY * mSizeX + inZ * mSizeX * mSizeY;
	}

	/// 读取体素的原始字节（0 = 空，非 0 = 实心）。越界一律返回 0。
	/// 上层可以把这个字节当作材质 / 调色板索引使用，所以这里刻意返回原值而不是"是否实心"。
	inline uint8			GetVoxel(int inX, int inY, int inZ) const
	{
		if (inX < 0 || inX >= mSizeX || inY < 0 || inY >= mSizeY || inZ < 0 || inZ >= mSizeZ)
			return 0;
		return mVoxels[GetVoxelIndex(inX, inY, inZ)];
	}

	/// 设置一个体素：0 = 空，非 0 = 实心。越界坐标会被忽略。
	///
	/// 这会**立即**改变碰撞几何 —— 窄相位每次查询都直接读体素数据，所以写入之后不需要重建形状或刚体。
	/// 但它**不会**更新质量属性，需要另外调用 Recalculate()。
	///
	/// 形状不拥有体素数据，所以这里写的是调用方的缓冲区（指针成员是 const 的，内部做一次 const_cast）。
	/// 这里也是 const 成员，与 Recalculate() 保持一致的语义：破坏系统可以在 const 形状上改体素。
	inline void				SetVoxel(int inX, int inY, int inZ, uint8 inValue) const
	{
		if (inX < 0 || inX >= mSizeX || inY < 0 || inY >= mSizeY || inZ < 0 || inZ >= mSizeZ)
			return;
		const_cast<uint8 *>(mVoxels)[GetVoxelIndex(inX, inY, inZ)] = inValue;
	}

	/// 该体素是否为实心（越界一律视为空）
	inline bool				IsSolid(int inX, int inY, int inZ) const
	{
		return GetVoxel(inX, inY, inZ) != 0;
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
	uint					GetNumSolidVoxels() const				{ return mNumSolidVoxels; }

	/// 表面体素的数量，即至少有一个空邻居的实心体素（缓存值）。
	/// 做"只遍历表面格"的优化时用这个数量做上界估计。
	uint					GetNumSurfaceVoxels() const				{ return mNumSurfaceVoxels; }

	/// 重新计算形状的缓存属性：实心体素数量、表面体素数量、实心体素的包围范围、体积、质量属性。
	/// 在就地修改体素数据之后必须调用；它只做一次线性扫描，比重建形状 + 重建刚体便宜得多。
	///
	/// 注意这个函数是 **const** 的：碰撞查询可以在 const 形状上刷新缓存，所以破坏系统不需要拿到非 const 形状。
	/// 为此下面所有缓存成员都是 mutable。
	void					Recalculate() const;

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
		return mMassProperties;
	}

	// 见 Shape::GetSubShapeIDBitsRecursive
	virtual uint			GetSubShapeIDBitsRecursive() const override
	{
		return mBitsX + mBitsY + mBitsZ;
	}

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
	/// 关：每格用单位盒（sGetUnitBox，凸半径 0），上游行为。
	///
	/// ★ **体素 vs 体素**不受它控制：sCollideVoxelGrids 永远走"跨格枚举格对筛选 + 整网格 SAT
	///   全局轴接触几何"这条单一路径（格子的球近似语义与这里的内切球一致）。
	///   CastRay / shape cast 一律仍按盒子处理（射线没有体积，用球会有洞；这条刻意分开，互不影响）。
	static bool				sUseSphereContacts;

private:
	// ---- 稀疏分块索引（8x8x8 一格 = 一个 64 位占用掩码）----
	//
	// 遍历"重叠区里的实心格"时，先看 chunk 的掩码：整块为空就一次跳过 512 个格子。
	// 掩码只占 1 bit / 512 体素（1/4096 的额外内存），所以它**不是**体素数据的副本 ——
	// 只是给体素图加的一层索引，和 Teardown 的 3D bitmap 是一个意思。
	static constexpr int	cChunkShift = 3;										///< 每个 chunk 每个轴 8 格
	static constexpr int	cChunkSize = 1 << cChunkShift;							///< 8
	static constexpr int	cChunkMask = cChunkSize - 1;							///< 7
	static constexpr uint	cChunkVolume = cChunkSize * cChunkSize * cChunkSize;	///< 512

	/// chunk 内部的线性下标（0..511）：lx + ly * 8 + lz * 64
	static inline uint		sGetChunkBitIndex(int inX, int inY, int inZ)
	{
		return uint((inX & cChunkMask) | ((inY & cChunkMask) << cChunkShift) | ((inZ & cChunkMask) << (2 * cChunkShift)));
	}

	/// 这个 chunk 里是否有实心格（越界一律 false）。整块为空时调用方可以一次跳过 512 个格子。
	inline bool				IsChunkSolid(int inChunkX, int inChunkY, int inChunkZ) const
	{
		if (inChunkX < 0 || inChunkX >= mNumChunks[0] || inChunkY < 0 || inChunkY >= mNumChunks[1] || inChunkZ < 0 || inChunkZ >= mNumChunks[2])
			return false;
		return mChunkMasks[inChunkX + inChunkY * mNumChunks[0] + inChunkZ * mNumChunks[0] * mNumChunks[1]] != 0;
	}

	/// 返回编码 inSize 个格子位置所需要的最小二进制位数（例如 10 个格子需要 4 位）
	static uint				sGetBitsForSize(int inSize);

	/// 手写的"体素网格 vs 体素网格"窄相位。
	///
	/// 这是 VoxelShape 性能和行为的关键路径，刻意**不走**碰撞分发表：
	/// 1. 先对两个网格的整体 OBB 做一次 SAT，求出唯一的最小分离轴（世界空间）—— 所有接触点共用这一根轴。
	///    ★ 这一步不能省：如果让每个格子各自算最小分离轴，深穿透（两个立方体大面积重合）时三轴会打平，
	///      每格挑到的轴都不一样，法线互相矛盾，求解器收到的净推进力抵消 ⇒ 表现为"两个方块叠在一起
	///      互相插着、推不开"。
/// 2. 再遍历两个网格重叠区里的实心格对（用分块掩码跳过空块）。格对用**跨格枚举**筛选：
///    把 shape1 格心映射到 shape2 的格坐标系，每轴取 floor(c-0.5-e) .. ceil(c+0.5+e)-1
///    （e = 全局轴方向上的推测余量，对齐时水平方向 e = 0）—— 不需要 AABB 变换，也不需要
///    格心距（LengthSq）筛选；把格子近似成内切球时这个集合是精确的（角点相碰、重叠面积为 0
///    的格对天然不在集合里）。每个候选产出 1 个点接触：法线/深度沿全局轴（与实心大盒同一根轴、
///    同一个公式），接触点 = 格心沿轴推到各自的格面；不填 mShape1Face/mShape2Face（面为空时
///    Jolt 退回单点接触）。
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
	/// 遍历按 8x8x8 的 chunk 推进：整块为空就一次跳过 512 个格子（见 IsChunkSolid），所以在一张大而稀疏的
	/// 网格上，开销只取决于"重叠区里真正有实心格的那几个 chunk"。
	template <class F>
	void					sVisitSolidVoxelsInBox(const AABox &inLocalBox, Vec3Arg inVoxelScale, Mat44Arg inVoxelTransform, const F &inVisitor) const;

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
	const uint8 *			mVoxels = nullptr;				///< 体素数据（不拥有所有权）
	float					mDensity = 1000.0f;				///< 密度（kg/m^3）
	RefConst<PhysicsMaterial> mMaterial;					///< 碰撞材质

	// ---- Recalculate() 计算出来的缓存（因为 Recalculate() 是 const，所以这里是 mutable）----
	mutable uint			mVersion = 0;					///< 每次 Recalculate() 递增，用来让调试几何失效
	mutable uint			mNumSolidVoxels = 0;			///< 实心体素数量
	mutable uint			mNumSurfaceVoxels = 0;			///< 表面体素数量（至少有一个空邻居的实心体素）
	mutable int				mSolidMin[3] = { 0, 0, 0 };		///< 实心体素包围范围的最小格子下标（X, Y, Z）
	mutable int				mSolidMax[3] = { -1, -1, -1 };	///< 实心体素包围范围的最大格子下标（闭区间；没有实心格时 max < min）
	mutable Vec3			mHalfExtent = Vec3::sZero();	///< 0.5 * size * voxelSize，形状局部空间里的半长
	mutable uint			mBitsX = 0;						///< 编码 X 坐标需要的 SubShapeID 位数
	mutable uint			mBitsY = 0;						///< 编码 Y 坐标需要的 SubShapeID 位数
	mutable uint			mBitsZ = 0;						///< 编码 Z 坐标需要的 SubShapeID 位数
	mutable MassProperties	mMassProperties;				///< 质量与转动惯量（转动惯量绕形状原点，即网格中心）

	// ---- 稀疏分块索引：每个 chunk 一个 64 位占用掩码（0 = 整块为空）----
	// 这层索引是 Recalculate() 算出来的，和体素数据一样会在就地编辑后过期；编辑体素后同样要调 Recalculate()。
	mutable int				mNumChunks[3] = { 0, 0, 0 };	///< 每个轴上的 chunk 数量
	mutable Array<uint64>	mChunkMasks;					///< 长度 = mNumChunks[0] * mNumChunks[1] * mNumChunks[2]

#ifdef JPH_DEBUG_RENDERER
	mutable DebugRenderer::GeometryRef	mGeometry;			///< 缓存的调试几何
	mutable uint			mGeometryVersion = uint(-1);	///< mGeometry 对应的 mVersion
#endif // JPH_DEBUG_RENDERER
};

JPH_NAMESPACE_END
