// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Physics/Collision/Shape/SubShapeID.h>
#include <Jolt/Physics/Collision/PhysicsMaterial.h>

#ifdef JPH_DEBUG_RENDERER
	#include <Jolt/Renderer/DebugRenderer.h>
#endif // JPH_DEBUG_RENDERER

JPH_NAMESPACE_BEGIN

class BoxShape;
class CollideShapeSettings;

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
/// 形状" UserConvex1）。窄相位把**每个实心体素**都当作一个独立的小盒子来处理，所以凹的体素数据会按"画出来的
/// 样子"参与碰撞 —— 墙上的窗户可以穿过去、挖出来的隧道是真正的隧道、倒塌的楼房会散架而不是变成一个实心大块。
/// 如果对体素求凸包（上一版实现），洞会被填平，上述行为全部失效。
///
/// 关键实现点：
///
/// 1. **没有派生几何**。形状不生成凸包也不生成三角形，窄相位遍历"两个形状包围盒重叠区"里的实心体素，
///    把每一格交给常规的"凸 vs 凸"通道（见 sCollideShapeVsVoxel）。
/// 2. **每格零额外内存**。所有格子共用同一个单位盒子（sGetUnitBox()，边长 1、中心在原点），每格只需要一次
///    "缩放 + 平移"，所以内存开销就等于体素数组本身（1 字节/格）。这也是体素碰撞体不能"每格生成一个 BoxShape"
///    的原因。
/// 3. **开销与网格尺寸解耦**。因为只遍历重叠区里的实心体素，所以与一面墙碰撞的开销和地图多大无关，只与
///    重叠区里的实心格数量有关（见 UnitTests 里的 VoxelShapeQueryScaling）。这让"大地图可破坏世界"成为可能。
/// 4. **SubShapeID 编码体素坐标**（见 EncodeSubShapeID / DecodeSubShapeID），破坏系统据此定位被击中的格子。
/// 5. **质量属性是缓存值**。碰撞查询读的是实时体素数据，但质量/惯量只在 Recalculate() 时才重算，所以
///    就地删体素后刚体的行为会立刻变化，而质量要等下一次 Recalculate() 才更新。
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

private:
	/// 返回编码 inSize 个格子位置所需要的最小二进制位数（例如 10 个格子需要 4 位）
	static uint				sGetBitsForSize(int inSize);

	/// 所有实心体素共享的单位盒子：边长 1、中心在原点、**凸半径 0**。
	/// 凸半径必须是 0，否则每格都会被撑大 cDefaultConvexRadius（默认 0.05 m），体素之间会互相"膨出"。
	static const BoxShape *	sGetUnitBox();

	/// 遍历 inLocalBox 覆盖到的所有**实心**体素，对每格调用一趟 inVisitor：
	///
	///		inVisitor(x, y, z, inCellTransform, inCellScale)
	///
	/// - inLocalBox：体素形状的**未缩放局部空间**（也就是 GetVoxelCenter 返回坐标的那个空间，单位：米）里的
	///   轴对齐盒。窄相位用它把遍历范围限制在"两个形状包围盒的重叠区"里。
	/// - inVoxelScale / inVoxelTransform：施加在体素形状上的缩放与质心变换，用来算格子的目标空间变换。
	/// - inCellTransform：把该格共享的"单位盒子"（边长 1、中心在原点）变换到目标空间（世界空间）的矩阵。
	/// - inCellScale：施加在该格单位盒子上的缩放，等于 inVoxelScale * mVoxelSize。
	///
	/// 这套"把格子的变换算好再交给访问者"的设计，是为了让所有窄相位回调都只关心"拿到这一格怎么用"，
	/// 不必各自重复一遍格心/缩放/SubShapeID 的换算。
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
	// 体素形状不做"整体一次凸查询"，而是遍历重叠区里的每一格，逐格走常规的凸-凸通道。
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

#ifdef JPH_DEBUG_RENDERER
	mutable DebugRenderer::GeometryRef	mGeometry;			///< 缓存的调试几何
	mutable uint			mGeometryVersion = uint(-1);	///< mGeometry 对应的 mVersion
#endif // JPH_DEBUG_RENDERER
};

JPH_NAMESPACE_END
