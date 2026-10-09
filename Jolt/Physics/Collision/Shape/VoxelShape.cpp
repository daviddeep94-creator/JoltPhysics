// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#include <Jolt/Physics/Collision/Shape/VoxelShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
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

template <class F>
void VoxelShape::sVisitSolidVoxelsInBox(const AABox &inLocalBox, Vec3Arg inVoxelScale, Mat44Arg inVoxelTransform, const F &inVisitor) const
{
	int min_x, min_y, min_z, max_x, max_y, max_z;
	if (!sGetCellRange(inLocalBox, min_x, min_y, min_z, max_x, max_y, max_z))
		return;

	// 施加在"格子单位盒子"上的缩放：形状缩放 * 体素边长
	Vec3 cell_scale = inVoxelScale * mVoxelSize;

	// 把格子单位盒子从体素局部空间搬到目标空间：目标变换 * 形状缩放 * 平移到格心
	Mat44 voxel_to_target = inVoxelTransform * Mat44::sScale(inVoxelScale);

	for (int z = min_z; z <= max_z; ++z)
		for (int y = min_y; y <= max_y; ++y)
			for (int x = min_x; x <= max_x; ++x)
				if (IsSolid(x, y, z))
					inVisitor(x, y, z, voxel_to_target * Mat44::sTranslation(GetVoxelCenter(x, y, z)), cell_scale);
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
			t_max[axis] = (boundary - o) / d;
			t_delta[axis] = s / d;
		}
		else if (d < -cDirectionEpsilon)
		{
			step[axis] = -1;
			float boundary = cell[axis] * s - extent[axis];			// 该格沿该轴的下边界
			t_max[axis] = (boundary - o) / d;
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
	voxel->sVisitSolidVoxelsInBox(local_box, inScale2, inCenterOfMassTransform2,
		[inShape1, inScale1, inCenterOfMassTransform1, &inSubShapeIDCreator1, voxel, &inSubShapeIDCreator2, &inCollideShapeSettings, &ioCollector, &inShapeFilter, unit_box](int inX, int inY, int inZ, const Mat44 &inCellTransform, Vec3 inCellScale)
		{
			// 该格的 SubShapeID：低若干位是 X，然后是 Y，最后是 Z。破坏系统据此知道该删哪一格。
			SubShapeIDCreator cell_id = voxel->EncodeSubShapeID(inSubShapeIDCreator2, uint(inX), uint(inY), uint(inZ));

			// 每格就是一个独立的"凸盒子 vs 凸盒子"查询，交给常规凸-凸通道处理
			CollisionDispatch::sCollideShapeVsShape(inShape1, unit_box, inScale1, inCellScale, inCenterOfMassTransform1, inCellTransform, inSubShapeIDCreator1, cell_id, inCollideShapeSettings, ioCollector, inShapeFilter);
		});
}

void VoxelShape::sCollideVoxelVsVoxel(const Shape *inShape1, const Shape *inShape2, Vec3Arg inScale1, Vec3Arg inScale2, Mat44Arg inCenterOfMassTransform1, Mat44Arg inCenterOfMassTransform2, const SubShapeIDCreator &inSubShapeIDCreator1, const SubShapeIDCreator &inSubShapeIDCreator2, const CollideShapeSettings &inCollideShapeSettings, CollideShapeCollector &ioCollector, const ShapeFilter &inShapeFilter)
{
	// 这个函数注册在 (体素, 任意形状) 上：shape1 一定是体素形状。
	// shape2 可能是另一个体素形状，也可能是普通的凸/网格形状 —— 两种情况都不用在这里区分：
	// 我们只遍历 shape1 的格子，把每一格和整个 shape2 交给通用分发；
	// 如果 shape2 也是体素，通用分发会自己再走一次 sCollideConvexVsVoxel 去遍历它的格子。
	const VoxelShape *voxel = static_cast<const VoxelShape *>(inShape1);

	// 见 sCollideConvexVsVoxel：同样要按 mMaxSeparationDistance 外扩，否则"靠近但未重叠"的接触会整片丢失。
	AABox shape2_world_bounds = inShape2->GetWorldSpaceBounds(inCenterOfMassTransform2, inScale2);
	shape2_world_bounds.ExpandBy(Vec3::sReplicate(inCollideShapeSettings.mMaxSeparationDistance));
	AABox local_box = sWorldBoxToVoxelLocal(shape2_world_bounds, inCenterOfMassTransform1, inScale1);

	const BoxShape *unit_box = sGetUnitBox();
	voxel->sVisitSolidVoxelsInBox(local_box, inScale1, inCenterOfMassTransform1,
		[inShape2, inScale2, inCenterOfMassTransform2, voxel, &inSubShapeIDCreator1, &inSubShapeIDCreator2, &inCollideShapeSettings, &ioCollector, &inShapeFilter, unit_box](int inX, int inY, int inZ, const Mat44 &inCellTransform, Vec3 inCellScale)
		{
			SubShapeIDCreator cell_id = voxel->EncodeSubShapeID(inSubShapeIDCreator1, uint(inX), uint(inY), uint(inZ));
			CollisionDispatch::sCollideShapeVsShape(unit_box, inShape2, inCellScale, inScale2, inCellTransform, inCenterOfMassTransform2, cell_id, inSubShapeIDCreator2, inCollideShapeSettings, ioCollector, inShapeFilter);
		});
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
