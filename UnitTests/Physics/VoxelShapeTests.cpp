// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include "UnitTestFramework.h"
#include "PhysicsTestContext.h"
#include "Layers.h"
#include <Jolt/Physics/Collision/Shape/VoxelShape.h>
#include <Jolt/Physics/Collision/Shape/MutableCompoundShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
#include <Jolt/Physics/Collision/Shape/PlaneShape.h>
#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/RayCast.h>
#include <Jolt/Physics/Collision/CastResult.h>
#include <Jolt/Physics/Collision/ShapeCast.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/Collision/CollisionDispatch.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>

#ifdef JPH_DEBUG_RENDERER
	#include <Jolt/Renderer/DebugRendererSimple.h>
#endif // JPH_DEBUG_RENDERER

JPH_SUPPRESS_WARNINGS_STD_BEGIN
#include <chrono>
#include <cmath>			// sqrtf
#include <cstdio>			// snprintf, used to build the benchmark names
JPH_SUPPRESS_WARNINGS_STD_END

TEST_SUITE("VoxelShapeTests")
{
	// All shapes in these tests are 16 x 16 x 16 voxels of 0.0625 m, so they have a bounding box of 1 x 1 x 1 meters.
	// One meter is 16 cells, i.e. a 0.5 m block is exactly one 8 x 8 x 8 chunk of cells.
	constexpr int cNumVoxels = 16;
	constexpr float cVoxelSize = 0.0625f;
	constexpr float cHalfExtent = 0.5f * cNumVoxels * cVoxelSize;

	/// Holds the voxel data and the shape that reads it (the shape does not own the data).
	///
	/// The data is the shared occupancy bitmap: 1 bit per voxel, one uint64 per 4x4x4 sub chunk
	/// (bit index = (x&3) | ((y&3)<<2) | ((z&3)<<4), which puts a row of 4 x's in one nibble).
	/// Cell() returns a proxy that packs a write into that word, so all the shape builders below
	/// can keep writing `Cell(x, y, z) = 1`.
	struct VoxelObject
	{
							VoxelObject(int inSizeX, int inSizeY, int inSizeZ, float inVoxelSize) :
								mSizeX(inSizeX), mSizeY(inSizeY), mSizeZ(inSizeZ), mVoxelSize(inVoxelSize)
		{
			mTexelsX = (inSizeX + 3) >> 2;
			mTexelsY = (inSizeY + 3) >> 2;
			mTexelPlane = mTexelsX * mTexelsY;
			mOccupancy.resize(size_t(mTexelPlane) * ((inSizeZ + 3) >> 2), 0);
			mShape = new VoxelShape(inSizeX, inSizeY, inSizeZ, inVoxelSize, mOccupancy.data());
		}

		/// A single voxel, as a writable reference into the occupancy bitmap (`word |= bit` / `word &= ~bit`)
		struct CellRef
		{
			uint64 &			mWord;
			uint64				mBit;

			CellRef &			operator = (uint8 inValue)	{ if (inValue != 0) mWord |= mBit; else mWord &= ~mBit; return *this; }
								operator bool () const		{ return (mWord & mBit) != 0; }
		};

		inline uint64 &		Word(int inX, int inY, int inZ)
		{
			return mOccupancy[(inX >> 2) + (inY >> 2) * mTexelsX + (inZ >> 2) * mTexelPlane];
		}

		inline uint64		Bit(int inX, int inY, int inZ) const
		{
			return uint64(1) << ((inX & 3) | ((inY & 3) << 2) | ((inZ & 3) << 4));
		}

		CellRef				Cell(int inX, int inY, int inZ)
		{
			JPH_ASSERT(inX >= 0 && inX < mSizeX && inY >= 0 && inY < mSizeY && inZ >= 0 && inZ < mSizeZ);
			return CellRef { Word(inX, inY, inZ), Bit(inX, inY, inZ) };
		}

		/// Whether a voxel is solid (reads back what Cell() wrote)
		bool				Get(int inX, int inY, int inZ) const
		{
			JPH_ASSERT(inX >= 0 && inX < mSizeX && inY >= 0 && inY < mSizeY && inZ >= 0 && inZ < mSizeZ);
			return (mOccupancy[(inX >> 2) + (inY >> 2) * mTexelsX + (inZ >> 2) * mTexelPlane] & Bit(inX, inY, inZ)) != 0;
		}

		/// Center of a voxel relative to the center of the grid
		Vec3				VoxelCenter(int inX, int inY, int inZ) const
		{
			return Vec3((inX + 0.5f) * mVoxelSize, (inY + 0.5f) * mVoxelSize, (inZ + 0.5f) * mVoxelSize)
				- Vec3(0.5f * mSizeX * mVoxelSize, 0.5f * mSizeY * mVoxelSize, 0.5f * mSizeZ * mVoxelSize);
		}

		int					CountSolid() const
		{
			int count = 0;
			for (int z = 0; z < mSizeZ; ++z)
				for (int y = 0; y < mSizeY; ++y)
					for (int x = 0; x < mSizeX; ++x)
						if (Get(x, y, z))
							++count;
			return count;
		}

		/// Refresh the cached properties of the shape (solid count, volume, mass properties) after the voxel data changed
		void				Update()
		{
			static_cast<const VoxelShape *>(mShape.GetPtr())->Recalculate();
		}

		int					mSizeX, mSizeY, mSizeZ;
		float				mVoxelSize;
		int					mTexelsX = 1, mTexelsY = 1, mTexelPlane = 1;
		Array<uint64>		mOccupancy;
		RefConst<Shape>		mShape;
	};

	/// Fill the entire grid (an axis aligned cube)
	static void sMakeCube(VoxelObject &ioObject)
	{
		for (int z = 0; z < ioObject.mSizeZ; ++z)
			for (int y = 0; y < ioObject.mSizeY; ++y)
				for (int x = 0; x < ioObject.mSizeX; ++x)
					ioObject.Cell(x, y, z) = 1;
		ioObject.Update();
	}

	/// Fill the voxels whose center is within inRadius of the center of the grid (a sphere with the grid center as origin)
	static void sMakeSphere(VoxelObject &ioObject, float inRadius)
	{
		for (int z = 0; z < ioObject.mSizeZ; ++z)
			for (int y = 0; y < ioObject.mSizeY; ++y)
				for (int x = 0; x < ioObject.mSizeX; ++x)
					if (ioObject.VoxelCenter(x, y, z).LengthSq() <= inRadius * inRadius)
						ioObject.Cell(x, y, z) = 1;
		ioObject.Update();
	}

	/// Fill a cylinder around the Y axis
	static void sMakeCylinder(VoxelObject &ioObject, float inRadius)
	{
		for (int z = 0; z < ioObject.mSizeZ; ++z)
			for (int y = 0; y < ioObject.mSizeY; ++y)
				for (int x = 0; x < ioObject.mSizeX; ++x)
				{
					Vec3 c = ioObject.VoxelCenter(x, y, z);
					if (c.GetX() * c.GetX() + c.GetZ() * c.GetZ() <= inRadius * inRadius)
						ioObject.Cell(x, y, z) = 1;
				}
		ioObject.Update();
	}

	/// A voxel cone: inSize cells both in diameter and in height, apex up along the local Y axis
	static void sMakeCone(VoxelObject &ioObject)
	{
		float size = float(ioObject.mSizeY);
		for (int z = 0; z < ioObject.mSizeZ; ++z)
			for (int y = 0; y < ioObject.mSizeY; ++y)
				for (int x = 0; x < ioObject.mSizeX; ++x)
				{
					float h = (float(y) + 0.5f) / size;
					float radius = 0.5f * size * (1.0f - h);
					float cx = float(x) + 0.5f - 0.5f * size;
					float cz = float(z) + 0.5f - 0.5f * size;
					if (cx * cx + cz * cz <= radius * radius)
						ioObject.Cell(x, y, z) = 1;
				}
		ioObject.Update();
	}

	/// A voxel torus around the local Y axis, tube center line radius inRing and tube radius inTube (in cells)
	static void sMakeTorus(VoxelObject &ioObject, float inRing, float inTube)
	{
		for (int z = 0; z < ioObject.mSizeZ; ++z)
			for (int y = 0; y < ioObject.mSizeY; ++y)
				for (int x = 0; x < ioObject.mSizeX; ++x)
				{
					float cx = float(x) + 0.5f - 0.5f * ioObject.mSizeX;
					float cy = float(y) + 0.5f - 0.5f * ioObject.mSizeY;
					float cz = float(z) + 0.5f - 0.5f * ioObject.mSizeZ;
					float dr = sqrt(cx * cx + cz * cz) - inRing;
					if (dr * dr + cy * cy <= inTube * inTube)
						ioObject.Cell(x, y, z) = 1;
				}
		ioObject.Update();
	}

	/// Make one of the 3 test shapes (0 = cube, 1 = sphere, 2 = cylinder)
	static void sMakeShape(VoxelObject &ioObject, int inShapeType)
	{
		switch (inShapeType)
		{
		case 0: sMakeCube(ioObject); break;
		case 1: sMakeSphere(ioObject, cHalfExtent); break;
		case 2: sMakeCylinder(ioObject, cHalfExtent); break;
		default: JPH_ASSERT(false);
		}
	}

	/// Do a narrow phase collide query between 2 shapes
	static void sCollide(const Shape *inShape1, Mat44Arg inTransform1, const Shape *inShape2, Mat44Arg inTransform2, const CollideShapeSettings &inSettings, AllHitCollisionCollector<CollideShapeCollector> &outCollector)
	{
		CollisionDispatch::sCollideShapeVsShape(inShape1, inShape2, Vec3::sOne(), Vec3::sOne(), inTransform1, inTransform2, SubShapeIDCreator(), SubShapeIDCreator(), inSettings, outCollector);
	}

	/// Check that a double (e.g. a body position) is close to an expected value
	static void sCheckApproxEqual(double inActual, double inExpected, double inTolerance = 1.0e-6)
	{
		CHECK(abs(inActual - inExpected) <= inTolerance);
	}

	/// A 10 x 10 flat mesh at y = 0 (2 triangles, so the shared edge is not an active edge)
	static RefConst<Shape> sCreateMeshFloor()
	{
		TriangleList triangles;
		triangles.push_back(Triangle(Vec3(-5, 0, -5), Vec3(-5, 0, 5), Vec3(5, 0, 5)));
		triangles.push_back(Triangle(Vec3(-5, 0, -5), Vec3(5, 0, 5), Vec3(5, 0, -5)));
		return MeshShapeSettings(triangles).Create().Get();
	}

	/// A 4 x 4 flat height field at y = 0
	static RefConst<Shape> sCreateHeightFieldFloor()
	{
		HeightFieldShapeSettings settings;
		settings.mSampleCount = 5;
		settings.mOffset = Vec3(-2, 0, -2);
		settings.mScale = Vec3(1, 1, 1);
		settings.mHeightSamples.resize(25, 0.0f);
		return settings.Create().Get();
	}

	TEST_CASE("VoxelShapeGeometry")
	{
		// A fully filled grid is a 1x1x1 cube of 1000 kg
		VoxelObject cube(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCube(cube);
		const VoxelShape *voxel_shape = static_cast<const VoxelShape *>(cube.mShape.GetPtr());

		CHECK(voxel_shape->GetSizeX() == cNumVoxels);
		CHECK(voxel_shape->GetNumSolidVoxels() == cNumVoxels * cNumVoxels * cNumVoxels);
		CHECK_APPROX_EQUAL(cube.mShape->GetLocalBounds().mMin, Vec3(-cHalfExtent, -cHalfExtent, -cHalfExtent));
		CHECK_APPROX_EQUAL(cube.mShape->GetLocalBounds().mMax, Vec3(cHalfExtent, cHalfExtent, cHalfExtent));
		CHECK_APPROX_EQUAL(cube.mShape->GetVolume(), 1.0f, 1.0e-5f);

		const MassProperties &cube_mp = cube.mShape->GetMassProperties();
		CHECK_APPROX_EQUAL(cube_mp.mMass, 1000.0f, 1.0e-3f);

		// Inertia of a solid 1x1x1 box of 1000 kg: m * (1^2 + 1^2) / 12 = 1000 / 6
		CHECK_APPROX_EQUAL(cube_mp.mInertia.GetColumn3(0).GetX(), 1000.0f / 6.0f, 0.1f);
		CHECK_APPROX_EQUAL(cube_mp.mInertia.GetColumn3(1).GetY(), 1000.0f / 6.0f, 0.1f);
		CHECK_APPROX_EQUAL(cube_mp.mInertia.GetColumn3(2).GetZ(), 1000.0f / 6.0f, 0.1f);
		CHECK_APPROX_EQUAL(cube_mp.mInertia.GetColumn3(0).GetY(), 0.0f, 1.0e-3f);

		// The sub shape ID encodes the voxel coordinate so that a collision result can be traced back to the
		// voxel that was hit (a destruction system needs to know which voxels to remove), it must round trip
		for (int z = 0; z < cNumVoxels; z += 3)
			for (int y = 0; y < cNumVoxels; y += 3)
				for (int x = 0; x < cNumVoxels; x += 3)
				{
					SubShapeID sub_id = voxel_shape->EncodeSubShapeID(SubShapeIDCreator(), uint(x), uint(y), uint(z)).GetID();
					uint dx, dy, dz;
					voxel_shape->DecodeSubShapeID(sub_id, dx, dy, dz);
					CHECK(dx == uint(x));
					CHECK(dy == uint(y));
					CHECK(dz == uint(z));
				}

		// A voxelized sphere of radius 0.5 has a volume of 4/3 pi r^3 = 0.5236, the discretization is within 10%
		VoxelObject sphere(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeSphere(sphere, cHalfExtent);
		CHECK_APPROX_EQUAL(sphere.mShape->GetVolume(), 4.0f / 3.0f * 3.14159265f * cHalfExtent * cHalfExtent * cHalfExtent, 0.06f);
		CHECK(sphere.CountSolid() < cNumVoxels * cNumVoxels * cNumVoxels); // The corners are empty

		// A sphere is symmetric so the inertia tensor is a multiple of the identity
		const MassProperties &sphere_mp = sphere.mShape->GetMassProperties();
		CHECK_APPROX_EQUAL(sphere_mp.mInertia.GetColumn3(0).GetX(), sphere_mp.mInertia.GetColumn3(2).GetZ(), 1.0e-3f);
		CHECK_APPROX_EQUAL(sphere_mp.mInertia.GetColumn3(0).GetX(), sphere_mp.mInertia.GetColumn3(1).GetY(), 1.0e-3f);

		// A voxelized cylinder of radius 0.5 has a volume of pi r^2 h = 0.7854
		VoxelObject cylinder(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCylinder(cylinder, cHalfExtent);
		CHECK_APPROX_EQUAL(cylinder.mShape->GetVolume(), 3.14159265f * cHalfExtent * cHalfExtent * 1.0f, 0.08f);

		// The inertia around the cylinder axis is smaller than around the other axes
		const MassProperties &cylinder_mp = cylinder.mShape->GetMassProperties();
		CHECK(cylinder_mp.mInertia.GetColumn3(1).GetY() < cylinder_mp.mInertia.GetColumn3(0).GetX());
	}

	/// Buoyancy. A voxel shape has no derived geometry, so GetSubmergedVolume walks the cells itself: one
	/// early out for "this cell is completely above the water", one for "completely below" and the
	/// tetrahedron clipping for the cells that straddle the plane. Those early outs are what makes a large
	/// grid cheap, so they must never swallow a straddling cell - and the clipping has to be done with a
	/// cube of the *real* voxel size: with a 1 m cube the submerged volume gets smeared over +-0.5 m, which
	/// keeps the total right for symmetric objects but puts the center of buoyancy in the wrong place.
	TEST_CASE("VoxelShapeSubmergedVolume")
	{
		const float one_cell = cVoxelSize * cVoxelSize * cVoxelSize;

		// Horizontal surface at inHeight, returns the submerged volume (total volume and center through out params)
		auto measure = [](const Shape *inShape, Mat44Arg inTransform, Vec3Arg inScale, float inHeight, float &outTotal, Vec3 &outCenter)
		{
			Plane surface = Plane::sFromPointAndNormal(Vec3(0, inHeight, 0), Vec3::sAxisY());
			float submerged;
			inShape->GetSubmergedVolume(inTransform, inScale, surface, outTotal, submerged, outCenter JPH_IF_DEBUG_RENDERER(, RVec3::sZero()));
			return submerged;
		};

		// An empty grid displaces nothing, no matter where the surface is
		{
			VoxelObject empty(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
			empty.Update();
			float total; Vec3 center;
			CHECK(measure(empty.mShape, Mat44::sIdentity(), Vec3::sOne(), 0.0f, total, center) == 0.0f);
			CHECK(total == 0.0f);
			CHECK(center == Vec3::sZero());
		}

		VoxelObject cube(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCube(cube);

		// The full grid is a 1 x 1 x 1 m cube (16 cells of 0.0625 m)
		float total; Vec3 center;

		// Surface completely below / completely above the cube
		CHECK(measure(cube.mShape, Mat44::sIdentity(), Vec3::sOne(), -10.0f, total, center) == 0.0f);
		CHECK_APPROX_EQUAL(total, 1.0f);
		CHECK_APPROX_EQUAL(measure(cube.mShape, Mat44::sIdentity(), Vec3::sOne(), 10.0f, total, center), 1.0f);
		CHECK_APPROX_EQUAL(center, Vec3(0, 0, 0));

		// Surface through the middle (which is a cell boundary): half the volume, center of buoyancy at -0.25
		CHECK_APPROX_EQUAL(measure(cube.mShape, Mat44::sIdentity(), Vec3::sOne(), 0.0f, total, center), 0.5f);
		CHECK_APPROX_EQUAL(center, Vec3(0, -0.25f, 0));

		// Surface at a quarter of the way up: 3/4 submerged, the centroid of [-0.5, 0.25] is at -0.125
		CHECK_APPROX_EQUAL(measure(cube.mShape, Mat44::sIdentity(), Vec3::sOne(), 0.25f, total, center), 0.75f);
		CHECK_APPROX_EQUAL(center, Vec3(0, -0.125f, 0));

		// A 90 degree rotation around X maps the cube onto itself, so every number must stay the same.
		// This is what pins down "cell center = origin + index * step" and the axis handling.
		{
			Mat44 rotated = Mat44::sRotation(Quat::sRotation(Vec3::sAxisX(), DegreesToRadians(90.0f)));
			CHECK_APPROX_EQUAL(measure(cube.mShape, rotated, Vec3::sOne(), 0.0f, total, center), 0.5f, 1.0e-4f);
			CHECK_APPROX_EQUAL(center, Vec3(0, -0.25f, 0), 1.0e-4f);
		}

		// Uniform scale 2: the total volume and the submerged volume scale with it (8x), the center too (-0.5)
		CHECK_APPROX_EQUAL(measure(cube.mShape, Mat44::sIdentity(), Vec3::sReplicate(2.0f), 0.0f, total, center), 4.0f, 1.0e-4f);
		CHECK_APPROX_EQUAL(total, 8.0f, 1.0e-4f);
		CHECK_APPROX_EQUAL(center, Vec3(0, -0.5f, 0), 1.0e-4f);

		// Non-uniform scale only along Y: a 1 x 2 x 1 box, half of it is 1 m^3
		CHECK_APPROX_EQUAL(measure(cube.mShape, Mat44::sIdentity(), Vec3(1, 2, 1), 0.0f, total, center), 1.0f, 1.0e-4f);
		CHECK_APPROX_EQUAL(total, 2.0f, 1.0e-4f);
		CHECK_APPROX_EQUAL(center, Vec3(0, -0.5f, 0), 1.0e-4f);

		// A single cell cut exactly through its center is the pure "straddles the surface" case: it must
		// produce half a cell, not a whole one
		{
			VoxelObject single(4, 4, 4, cVoxelSize);
			single.Cell(0, 0, 0) = 1;
			single.Update();

			Vec3 cell_center = single.VoxelCenter(0, 0, 0);
			float single_total; Vec3 single_center;
			CHECK_APPROX_EQUAL(measure(single.mShape, Mat44::sIdentity(), Vec3::sOne(), cell_center.GetY(), single_total, single_center), 0.5f * one_cell, 1.0e-7f);
			CHECK_APPROX_EQUAL(single_total, one_cell, 1.0e-7f);
			CHECK_APPROX_EQUAL(single_center, Vec3(cell_center.GetX(), cell_center.GetY() - 0.25f * cVoxelSize, cell_center.GetZ()), 1.0e-7f);
		}

		// Raising the surface must monotonically increase the submerged volume, and it can never exceed the
		// total volume (this sweeps the surface through the whole grid, so both early outs and the clipping
		// path are exercised on a shape that is not a box)
		{
			VoxelObject sphere(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
			sMakeSphere(sphere, cHalfExtent);

			float previous = 0.0f;
			for (int i = 0; i <= 16; ++i)
			{
				float submerged = measure(sphere.mShape, Mat44::sIdentity(), Vec3::sOne(), -0.6f + 0.075f * float(i), total, center);
				CHECK(submerged >= previous - 1.0e-6f);
				CHECK(submerged <= total + 1.0e-6f);
				previous = submerged;
			}
			CHECK(previous > 0.0f);
		}
	}

	TEST_CASE("VoxelShapeVsMeshShape")
	{
		RefConst<Shape> mesh = sCreateMeshFloor();
		CHECK(mesh != nullptr);
		if (mesh == nullptr)
			return;
		CHECK(mesh->GetSubType() == EShapeSubType::Mesh);

		VoxelObject cube(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCube(cube);

		// The cube is pushed a quarter cell into the floor. The push in has to stay below half a cell:
		// the collision geometry of a cell is the inscribed sphere at its center, so pushing in by half
		// a cell or more puts the centers below the plane of the floor triangles, and the mesh query
		// (mBackFaceMode defaults to IgnoreBackFaces) then treats the floor as back facing and reports
		// nothing at all. With the centers above the plane the reported depth is exactly the push in.
		{
			const float cPushIn = 0.25f * cVoxelSize;
			CollideShapeSettings settings;
			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(cube.mShape, Mat44::sTranslation(Vec3(0, cHalfExtent - cPushIn, 0)), mesh, Mat44::sIdentity(), settings, collector);
			Trace("mesh floor, pushed in %.5f: %d hits", cPushIn, int(collector.mHits.size()));
			CHECK(collector.mHits.size() > 0);
			if (!collector.mHits.empty())
			{
				CHECK_APPROX_EQUAL(collector.mHits[0].mPenetrationDepth, cPushIn, 0.002f);
				CHECK(abs(collector.mHits[0].mPenetrationAxis.Normalized().GetY()) > 0.99f);
			}
		}

		// Just above the floor (0.03 m apart): with a max separation distance we get a hit with a negative depth
		{
			CollideShapeSettings settings;
			settings.mMaxSeparationDistance = 0.1f;
			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(cube.mShape, Mat44::sTranslation(Vec3(0, cHalfExtent + 0.03f, 0)), mesh, Mat44::sIdentity(), settings, collector);
			CHECK(collector.mHits.size() > 0);
			if (!collector.mHits.empty())
				CHECK_APPROX_EQUAL(collector.mHits[0].mPenetrationDepth, -0.03f, 0.002f);
		}

		// A ray from above hits the top face of the cube
		{
			RayCastResult hit;
			CHECK(cube.mShape->CastRay(RayCast(Vec3(0, 2, 0), Vec3(0, -2, 0)), SubShapeIDCreator(), hit));
			if (hit.mFraction < 1.0f)
				CHECK_APPROX_EQUAL(hit.mFraction, (2.0f - cHalfExtent) / 2.0f, 0.01f);
		}
	}

	TEST_CASE("VoxelShapeVsVoxelShape")
	{
		VoxelObject cube1(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCube(cube1);
		VoxelObject cube2(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCube(cube2);

		// Stacked with an overlap of 0.05
		{
			CollideShapeSettings settings;
			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(cube1.mShape, Mat44::sTranslation(Vec3(0, cHalfExtent, 0)), cube2.mShape, Mat44::sTranslation(Vec3(0, 2.0f * cHalfExtent + cHalfExtent - 0.05f, 0)), settings, collector);
			CHECK(collector.mHits.size() > 0);
			if (!collector.mHits.empty())
			{
				const CollideShapeResult &hit = collector.mHits[0];
				CHECK_APPROX_EQUAL(hit.mPenetrationDepth, 0.05f, 0.002f);
				CHECK(abs(hit.mPenetrationAxis.Normalized().GetY()) > 0.99f);

				// The narrow phase emits point contacts: no contact faces (Jolt falls back to
				// mContactPointOn1/2 in ManifoldBetweenTwoFaces)
				CHECK(hit.mShape1Face.size() == 0);
				CHECK(hit.mShape2Face.size() == 0);
			}
		}

		// With no speculative contact distance the sphere-center filter keeps exactly the vertical
		// cell pairs of the 1 m^2 contact plane (the two grids share a lattice, so a lateral neighbour
		// sits sqrt(cell^2 + (0.2 * cell)^2) away, just above the cell threshold), each with the depth
		// and the axis that the convex hull of the whole grid used to report for the entire face.
		{
			CollideShapeSettings settings;
			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(cube1.mShape, Mat44::sTranslation(Vec3(0, cHalfExtent, 0)), cube2.mShape, Mat44::sTranslation(Vec3(0, 2.0f * cHalfExtent + cHalfExtent - 0.05f, 0)), settings, collector);
			CHECK(collector.mHits.size() == cNumVoxels * cNumVoxels);
			for (const CollideShapeResult &h : collector.mHits)
			{
				CHECK_APPROX_EQUAL(h.mPenetrationDepth, 0.05f, 0.002f);
				CHECK(abs(h.mPenetrationAxis.Normalized().GetY()) > 0.99f);
				CHECK(h.mShape1Face.size() == 0);
				CHECK(h.mShape2Face.size() == 0);
			}
		}

		// A voxel cube dropped on a static voxel cube comes to rest on top of it without tipping over
		{
			PhysicsTestContext c;
			c.CreateBody(BodyCreationSettings(cube1.mShape, RVec3(0, cHalfExtent, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

			BodyInterface &bi = c.GetBodyInterface();
			BodyID top = bi.CreateAndAddBody(BodyCreationSettings(cube2.mShape, RVec3(0, 1.6, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

			c.Simulate(1.0f);

			// The top cube must end up exactly on top of the bottom one (the top of the bottom cube is at y = 1)
			Trace("stack rest y=%.7f target=%.7f", bi.GetPosition(top).GetY(), 3.0 * cHalfExtent);
			sCheckApproxEqual(bi.GetPosition(top).GetY(), 3.0 * cHalfExtent, 0.02);
			CHECK(abs(bi.GetLinearVelocity(top).GetY()) < 0.01f);

			// It should not have rotated (with a single contact point it would tip over)
			Quat rotation = bi.GetRotation(top);
			CHECK(abs(rotation.GetX()) < 0.01f);
			CHECK(abs(rotation.GetY()) < 0.01f);
			CHECK(abs(rotation.GetZ()) < 0.01f);
		}
	}

	TEST_CASE("VoxelShapeSurfaceDebug")
	{
		VoxelObject cube1(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCube(cube1);
		VoxelObject cube2(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCube(cube2);

		const VoxelShape *s1 = static_cast<const VoxelShape *>(cube1.mShape.GetPtr());
		s1->GetNumSolidVoxels();	// EnsureDerived
		int rng[6] = { 0, cNumVoxels - 2, 0, cNumVoxels - 1, cNumVoxels - 1, cNumVoxels - 1 };
		int rng_all[6] = { 0, 0, 0, cNumVoxels - 1, cNumVoxels - 1, cNumVoxels - 1 };
		Trace("SurfaceDebug: solid=%u surfCount(top slab)=%d surfCount(all)=%d",
			s1->GetNumSolidVoxels(), s1->DebugCountSurfaceInRange(rng), s1->DebugCountSurfaceInRange(rng_all));

		int visited = s1->DebugVisitSurfaceBox(AABox(Vec3(-cHalfExtent, cHalfExtent - cVoxelSize - 1e-3f, -cHalfExtent),
			Vec3(cHalfExtent, cHalfExtent + 1e-3f, cHalfExtent)));
		Trace("SurfaceDebug: surface-only visit of top slab = %d", visited);
	}

	/// The scenario that motivated the center-distance pair filter: a wide voxel slab dropped on a voxel
	/// floor. Without the filter the candidate enumeration admits ~7.84 contacts per cell of the contact
	/// face (AABB overlap admits corner-touching cells), so a 30 x 30 cell contact face needs
	/// ~7000 contacts and cMaxVoxelContacts = 1024 truncates the list. Because the traversal order is
	/// z -> y -> x, the surviving contacts are all on the low z side, so the slab is supported on one
	/// edge only: it tips over and slides off ("the collider disappeared for a frame, the object sinks
	/// and is pushed back up"). With the filter the whole face is supported.
	TEST_CASE("VoxelShapeSphereModeSlabOnVoxelFloor")
	{
		constexpr int cFloorSize = 40;			// 2.5 x 2.5 m, 0.25 m thick
		constexpr int cSlabSize = 30;			// 1.875 m contact face
		constexpr int cSlabHeight = 4;			// 0.25 m thick

		VoxelObject floor(cFloorSize, 4, cFloorSize, cVoxelSize);
		sMakeCube(floor);
		VoxelObject slab(cSlabSize, cSlabHeight, cSlabSize, cVoxelSize);
		sMakeCube(slab);

		PhysicsTestContext c;

		// Floor: 0.25 m thick, top face at y = 0
		c.CreateBody(BodyCreationSettings(floor.mShape, RVec3(0, -2.0f * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

		BodyInterface &bi = c.GetBodyInterface();
		BodyID id = bi.CreateAndAddBody(BodyCreationSettings(slab.mShape, RVec3(0, 0.3, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

		c.Simulate(3.0f);

		RVec3 p = bi.GetPosition(id);
		Quat r = bi.GetRotation(id);
		float drift = Vec3(float(p.GetX()), 0.0f, float(p.GetZ())).Length();
		float up_y = (r * Vec3::sAxisY()).GetY();

		Trace("slab on voxel floor: y %.4f drift %.4f up.y %.4f", p.GetY(), drift, up_y);

		// The slab stays on the floor, level and in place. It rests with its center at half its
		// thickness above the top face of the floor.
		sCheckApproxEqual(p.GetY(), 0.5 * cSlabHeight * cVoxelSize, 0.05);
		CHECK(drift < 0.05f);
		CHECK(up_y > 0.999f);					// Level to within ~2.6 degrees
	}

	/// Regression for the "stacked voxel blocks sink into each other / the flat contact plane turns into
	/// a bumpy wave" report. A dynamic slab rests on a static slab, offset by half a voxel in X.
	/// With a half-voxel offset the inscribed spheres only touch after ~0.134 voxel of sink, but the
	/// contact geometry against the global separation axis must push the slab back out to the
	/// face-to-face resting height, so the final sink must stay within the solver's penetration slop.
	TEST_CASE("VoxelShapeSphereModeStackHalfVoxelOffset")
	{
		constexpr int cBottomSize = 40;			// 2.5 x 2.5 m, 0.25 m thick
		constexpr int cTopSize = 30;			// 1.875 m contact face
		constexpr int cTopHeight = 4;			// 0.25 m thick

		VoxelObject bottom(cBottomSize, 4, cBottomSize, cVoxelSize);
		sMakeCube(bottom);
		VoxelObject top(cTopSize, cTopHeight, cTopSize, cVoxelSize);
		sMakeCube(top);

		PhysicsTestContext c;

		// Static bottom slab, top face at y = 0
		c.CreateBody(BodyCreationSettings(bottom.mShape, RVec3(0, -2.0f * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

		// Dynamic slab, offset half a voxel in X, dropped from just above the bottom slab
		BodyInterface &bi = c.GetBodyInterface();
		BodyID id = bi.CreateAndAddBody(BodyCreationSettings(top.mShape, RVec3(0.5f * cVoxelSize, 0.3, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

		c.Simulate(3.0f);

		RVec3 p = bi.GetPosition(id);
		Quat r = bi.GetRotation(id);
		float up_y = (r * Vec3::sAxisY()).GetY();
		float drift = Vec3(float(p.GetX()) - 0.5f * cVoxelSize, 0.0f, float(p.GetZ())).Length();

		Trace("half-offset stack: y %.4f drift %.4f up.y %.4f", p.GetY(), drift, up_y);

		// Resting height: center at half its thickness above the top face of the bottom slab.
		// The snap must push it back out to the face plane, so only the penetration slop may be missing.
		sCheckApproxEqual(p.GetY(), 0.5 * cTopHeight * cVoxelSize, 0.04);
		CHECK(drift < 0.05f);							// No sideways push from diagonal sphere normals
		CHECK(up_y > 0.999f);							// Level
	}

	/// Regression for the "two stacked voxel blocks get jammed" report (screenshot: the bottom layer of
	/// the upper block interleaves with the top layer of the lower block). Spawned half a voxel deep
	/// inside the floor and half a voxel offset in X, per-pair snapped normals used to point both
	/// up and down (buried interior contacts), cancelling out: the blocks stayed wedged. The single
	/// global separation axis keeps all contact normals consistent, so the block is pushed back out.
	TEST_CASE("VoxelShapeSphereModeJammedInterleave")
	{
		constexpr int cCubeSize = 10;			// 0.625 m cubes (both dynamic, like a drop test)
		constexpr int cFloorSizeJ = 40;			// 2.5 x 2.5 m, 0.25 m thick

		VoxelObject cube1(cCubeSize, cCubeSize, cCubeSize, cVoxelSize);
		sMakeCube(cube1);
		VoxelObject cube2(cCubeSize, cCubeSize, cCubeSize, cVoxelSize);
		sMakeCube(cube2);

		PhysicsTestContext c;

		// Static voxel floor, top face at y = 0
		VoxelObject floor(cFloorSizeJ, 4, cFloorSizeJ, cVoxelSize);
		sMakeCube(floor);
		c.CreateBody(BodyCreationSettings(floor.mShape, RVec3(0, -2.0f * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

		BodyInterface &bi = c.GetBodyInterface();

		// Lower dynamic cube resting on the floor
		BodyID id1 = bi.CreateAndAddBody(BodyCreationSettings(cube1.mShape, RVec3(0, 0.5f * cCubeSize * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

		// Upper dynamic cube spawned wedged a full voxel deep into the lower one (one voxel below
		// the face-to-face resting height of 1.5 cubes) and half a voxel off in X: the bottom layer of
		// the upper cube interleaves with the top layer of the lower cube. Without the exposure filter
		// the buried contacts push both up and down, cancelling out: the cubes stay wedged.
		BodyID id2 = bi.CreateAndAddBody(BodyCreationSettings(cube2.mShape, RVec3(0.5f * cVoxelSize, 1.5f * cCubeSize * cVoxelSize - cVoxelSize, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

		c.Simulate(3.0f);

		RVec3 p1 = bi.GetPosition(id1);
		RVec3 p2 = bi.GetPosition(id2);
		Quat r2 = bi.GetRotation(id2);
		float up_y2 = (r2 * Vec3::sAxisY()).GetY();

		Trace("jammed interleave: lower y %.4f upper y %.4f up.y %.4f", p1.GetY(), p2.GetY(), up_y2);

			// The upper cube must climb out of the wedge and rest on top of the lower one (1.5 cubes,
			// minus penetration slop); the lower cube must stay on the floor.
			sCheckApproxEqual(p1.GetY(), 0.5 * cCubeSize * cVoxelSize, 0.06);
			CHECK(p2.GetY() > 1.5f * cCubeSize * cVoxelSize - 0.06f);
			CHECK(p2.GetY() < 1.5f * cCubeSize * cVoxelSize + 0.06f);
			CHECK(up_y2 > 0.999f);
	}

	/// Force the iterate side of the voxel vs voxel narrow phase (see VoxelShape::sIterateSideOverride)
	/// for the duration of a scope. It is a process wide switch, so a test must restore it.
	struct IterateSideScope
	{
		IterateSideScope(int inSide) :
			mOld(VoxelShape::sIterateSideOverride)
		{
			VoxelShape::sIterateSideOverride = inSide;
		}

		~IterateSideScope()
		{
			VoxelShape::sIterateSideOverride = mOld;
		}

		int				mOld;
	};

	/// Force the interior pair filter (see VoxelShape::sInteriorSkipPairs; true = mode 1 "both interior")
	/// for the duration of a scope.
	struct InteriorPairFilterScope
	{
		InteriorPairFilterScope(bool inSkip) :
			mOld(VoxelShape::sInteriorSkipPairs)
		{
			VoxelShape::sInteriorSkipPairs = inSkip? 1 : 0;
		}

		~InteriorPairFilterScope()
		{
			VoxelShape::sInteriorSkipPairs = mOld;
		}

		int				mOld;
	};

	/// Force the contact normal source (see VoxelShape::sNormalMode) for the duration of a scope.
	struct NormalModeScope
	{
		NormalModeScope(int inMode) :
			mOld(VoxelShape::sNormalMode)
		{
			VoxelShape::sNormalMode = inMode;
		}

		~NormalModeScope()
		{
			VoxelShape::sNormalMode = mOld;
		}

		int				mOld;
	};

	/// Diagnostic dump used to choose the voxel vs voxel contact normal rule (see VoxelShape::sNormalMode).
	/// Runs the same query in four modes: 0 (the whole grid SAT axis), 1 (the per-cell-pair center line,
	/// i.e. the "sphere" rule), 2 (the exposed face normal) and 3 (the fusion: edge cells take 1, flat and
	/// embedded cells take 0, i.e. the global axis). It reports how far the per-pair normals deviate from
	/// the SAT axis, plus the best / second best overlap of the grid SAT (a candidate criterion for "flat
	/// (face or slant) contact" vs "deep interpenetration").
	///
	/// Three scenarios, picked because the two rules need opposite things:
	///   A) a voxel torus resting on a voxel cone - curved and slanted, the SAT axis has nothing to do
	///      with the local contact surface (the donut gets pushed straight up by the cone),
	///   B) a slab resting on a floor, half a voxel off in X - an aligned contact, the SAT axis is exact,
	///   C) two blocks wedged a full voxel deep with a half voxel lateral shift - deep, the per-pair
	///      normals contradict each other (this is the case that used to jam at y = 1.387).
	/// The numbers are printed, not asserted: they are the input for picking the rule.
	TEST_CASE("VoxelShapeNormalDiagnostics")
	{
		constexpr float cell = cVoxelSize;

		// Run one query in the given normal mode. inAxis is the reference direction (the mode 0 SAT axis);
		// it is only needed for the angle histogram, so the first call can pass anything.
		auto run_mode = [](int inMode, const VoxelObject &inA, Mat44Arg inT1, const VoxelObject &inB, Mat44Arg inT2, Vec3Arg inAxis,
			Vec3 &outAxis, float &outOverlap, float &outSecond, int &outCount, double &outMean, int (&outBuckets)[5])
		{
			NormalModeScope scope(inMode);
			CollideShapeSettings settings;
			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(inA.mShape, inT1, inB.mShape, inT2, settings, collector);

			outCount = int(collector.mHits.size());
			outAxis = collector.mHits.empty()? inAxis : collector.mHits[0].mPenetrationAxis.Normalized();
			outOverlap = VoxelShape::sLastGridOverlap[0];
			outSecond = VoxelShape::sLastGridOverlap[1];

			double sum = 0.0;
			for (int i = 0; i < 5; ++i)
				outBuckets[i] = 0;
			for (const CollideShapeResult &h : collector.mHits)
			{
				float d = abs(h.mPenetrationAxis.Normalized().Dot(inAxis));
				sum += d;
				if (d > 0.999f) ++outBuckets[0];
				else if (d > 0.99f) ++outBuckets[1];
				else if (d > 0.9f) ++outBuckets[2];
				else if (d > 0.7f) ++outBuckets[3];
				else ++outBuckets[4];
			}
			outMean = outCount > 0? sum / outCount : 0.0;
		};

		auto dump = [&](const char *inName, const VoxelObject &inA, Mat44Arg inT1, const VoxelObject &inB, Mat44Arg inT2)
		{
			Vec3 a[4];
			float overlap[4], second[4];
			int count[4], buckets[4][5];
			double mean[4];

			run_mode(0, inA, inT1, inB, inT2, Vec3::sAxisY(), a[0], overlap[0], second[0], count[0], mean[0], buckets[0]);
			for (int mode = 1; mode <= 3; ++mode)
				run_mode(mode, inA, inT1, inB, inT2, a[0], a[mode], overlap[mode], second[mode], count[mode], mean[mode], buckets[mode]);

			Trace("%-22s SAT best %6.3f second %6.3f ratio %.3f | mean |n.axis0|: m0 %.3f(%d) m1 %.3f(%d) m2 %.3f(%d) m3 %.3f(%d)",
				inName, overlap[0], second[0], second[0] > 0.0f? overlap[0] / second[0] : 0.0f,
				mean[0], count[0], mean[1], count[1], mean[2], count[2], mean[3], count[3]);
			Trace("    buckets >0.999 / >0.99 / >0.9 / >0.7 / rest   m2: %d %d %d %d %d    m3: %d %d %d %d %d",
				buckets[2][0], buckets[2][1], buckets[2][2], buckets[2][3], buckets[2][4],
				buckets[3][0], buckets[3][1], buckets[3][2], buckets[3][3], buckets[3][4]);

			CHECK(count[0] >= 0);
		};

		// A) A voxel torus (ring 7 cells, tube 3 cells -> a 20 x 7 x 20 grid) resting on a voxel cone
		// (15 cells diameter and height). Same numbers as Samples/Tests/Shapes/VoxelShapeTest.cpp.
		{
			VoxelObject cone(15, 15, 15, cell);
			sMakeCone(cone);
			VoxelObject donut(20, 7, 20, cell);
			sMakeTorus(donut, 7.0f, 3.0f);

			// The slant goes through (radial = 7.5, height = 0) and (radial = 0, height = 15), so the
			// tube center line (radial = 7) is cDonutTube away from it at height rest_height.
			constexpr float cConeRadius = 7.5f;
			float slant_length = sqrt(15.0f * 15.0f + cConeRadius * cConeRadius);
			float rest_height = (cConeRadius * 15.0f + 3.0f * slant_length - 15.0f * 7.0f) / cConeRadius;

			// Cone: base on the floor at y = 0, so its grid center is half its height up
			Mat44 cone_transform = Mat44::sTranslation(Vec3(0, 0.5f * 15.0f * cell, 0));
			for (float drop : { 0.0f, 0.5f, 1.0f })
			{
				char name[64];
				snprintf(name, sizeof(name), "A) donut on cone %+.1f", (double)drop);
				dump(name, donut, Mat44::sTranslation(Vec3(0, (rest_height + drop) * cell, 0)), cone, cone_transform);
			}
		}

		// B) A 1 m cube resting on a 1 m floor with a half voxel lateral offset (aligned contact).
		{
			VoxelObject floor(cNumVoxels, cNumVoxels, cNumVoxels, cell);
			sMakeCube(floor);
			VoxelObject box(cNumVoxels, cNumVoxels, cNumVoxels, cell);
			sMakeCube(box);

			dump("B) half offset stack", box, Mat44::sTranslation(Vec3(0.5f * cell, 0.5f * cNumVoxels * cell, 0)),
				floor, Mat44::sTranslation(Vec3(0, 1.5f * cNumVoxels * cell - 0.05f, 0)));
		}

		// C) Two 1 m cubes wedged a full voxel deep with a half voxel lateral shift (deep interpenetration,
		// same numbers as VoxelShapeSphereModeJammedInterleave)
		{
			VoxelObject lower(cNumVoxels, cNumVoxels, cNumVoxels, cell);
			sMakeCube(lower);
			VoxelObject upper(cNumVoxels, cNumVoxels, cNumVoxels, cell);
			sMakeCube(upper);

			dump("C) deep interleave", lower, Mat44::sTranslation(Vec3(0, 0.5f * cNumVoxels * cell, 0)),
				upper, Mat44::sTranslation(Vec3(0.5f * cell, 1.5f * cNumVoxels * cell - cell, 0)));
		}

		// D) Two 1 m cubes overlapping 50 percent in X and Y and fully in Z. This is the case the
		// best / second overlap ratio cannot tell apart from A: the overlap region is a 3D volume, so the
		// ratio is ~1, yet the per-pair normals are all over the place and the global axis is the right
		// answer. It is here to show that the ratio is not a sound discriminator.
		{
			VoxelObject lower(cNumVoxels, cNumVoxels, cNumVoxels, cell);
			sMakeCube(lower);
			VoxelObject upper(cNumVoxels, cNumVoxels, cNumVoxels, cell);
			sMakeCube(upper);

			dump("D) 3D half overlap", lower, Mat44::sTranslation(Vec3(0, 0.5f * cNumVoxels * cell, 0)),
				upper, Mat44::sTranslation(Vec3(0.5f * cNumVoxels * cell, 1.0f * cNumVoxels * cell, 0)));
		}

		// E2) The same 3D half overlap as D, but offset by half a cell instead of five cells in X. Here the
		// per-pair normals are NOT along the separation axis any more (the straddle window is one cell wide
		// along the axis, so the pairs sit at half a cell laterally), which makes it the annoying case for
		// any "use per-pair normals when the overlap is thick" rule.
		{
			VoxelObject lower(cNumVoxels, cNumVoxels, cNumVoxels, cell);
			sMakeCube(lower);
			VoxelObject upper(cNumVoxels, cNumVoxels, cNumVoxels, cell);
			sMakeCube(upper);

			dump("E2) deep + half offset", lower, Mat44::sTranslation(Vec3(0, 0.5f * cNumVoxels * cell, 0)),
				upper, Mat44::sTranslation(Vec3(0.5f * cell, 0.9f * cNumVoxels * cell, 0)));
		}

		// E3) A 1 m cube resting on a 2.5 m x 2.5 m voxel floor (the normal gameplay case). Thin contact,
		// huge bounding boxes: the SAT axis is exact here.
		{
			VoxelObject floor(40, 4, 40, cell);
			sMakeCube(floor);
			VoxelObject box(cNumVoxels, cNumVoxels, cNumVoxels, cell);
			sMakeCube(box);

			dump("E3) cube on voxel floor", box, Mat44::sTranslation(Vec3(0.5f * cell, 0.45f, 0)),
				floor, Mat44::sTranslation(Vec3(0, -2.0f * cell, 0)));
		}

		// F) The acceptance test for the donut report: the torus dropped on the cone, offset sideways so
		// that it is not perfectly centered on the cone (that is the case the report is about), simulated
		// once per normal mode. Reported: the resting height, the lateral drift (how far it slid off the
		// cone) and the tilt of its up axis.
		{
			VoxelObject cone(15, 15, 15, cell);
			sMakeCone(cone);
			VoxelObject donut(20, 7, 20, cell);
			sMakeTorus(donut, 7.0f, 3.0f);

			constexpr float cConeRadius = 7.5f;
			float slant_length = sqrt(15.0f * 15.0f + cConeRadius * cConeRadius);
			float rest_height = (cConeRadius * 15.0f + 3.0f * slant_length - 15.0f * 7.0f) / cConeRadius;

			float rest_y[4][4];
			for (int mode = 0; mode <= 3; ++mode)
				for (int i = 0; i < 4; ++i)
				{
					float offset = float(i);

					NormalModeScope scope(mode);

					PhysicsTestContext c;
					c.CreateBody(BodyCreationSettings(cone.mShape, RVec3(0, 0.5f * 15.0f * cell, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

					BodyInterface &bi = c.GetBodyInterface();
					BodyID id = bi.CreateAndAddBody(BodyCreationSettings(donut.mShape, RVec3(offset * cell, (rest_height + 2.0f) * cell, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

					c.Simulate(3.0f);

					RVec3 p = bi.GetPosition(id);
					Quat r = bi.GetRotation(id);
					float up_y = (r * Vec3::sAxisY()).GetY();
					rest_y[mode][i] = float(p.GetY());
					Trace("F) donut mode %d offset %.1f cells: y %.4f dx %+.4f dz %+.4f up.y %.3f",
						mode, (double)offset, p.GetY(), p.GetX() - offset * cell, p.GetZ(), up_y);
				}

			// This is the donut report: with the whole grid SAT axis the cone can only push the donut
			// straight up, so as soon as it is not perfectly centered it slides out and falls off
			// (y goes strongly negative, the trace shows +4 m of lateral drift and a tilted up axis).
			// The per-pair center line follows the cone slant, which is what forms the groove, and the
			// donut stays seated; the exposed face normal reaches the same thing through the stair steps.
			//
			// Caveat measured for sEdgeEitherSide == 1: the donut's own underside cells are flat, so a
			// criterion that looks at both cells lets the CONE's cells (a 45 degree slant, every cell of
			// which has two exposed faces) pull them onto the center line as well. The trace then shows
			// the donut rocking on the cone at offsets 2 and 3 (up.y 0.934 / 0.852 instead of 1.000, up
			// to 10 cm of lateral drift) - it still stays seated, hence the assertions below, but it is
			// no longer the "stable, no jitter" behaviour the fusion was picked for. With the criterion
			// judging the iterated side only (sEdgeEitherSide == 0) all four offsets stay upright
			// (up.y 1.000 / 0.982 / 1.000 / 0.999, y 0.9207 / 0.8881 / 0.9882 / 1.0055).
			for (int i = 1; i < 4; ++i)
			{
				CHECK(rest_y[0][i] < 0.0f);				// global axis: pushed off the cone and fell
				CHECK(rest_y[1][i] > 8.0f * cell);		// per-pair center line: stays seated on the cone
				CHECK(rest_y[2][i] > 8.0f * cell);		// exposed face normal: stays seated on the cone
				CHECK(rest_y[3][i] > 8.0f * cell);		// fusion (edge cells take the center line)
			}
		}
	}

	/// The narrow phase iterates the side that has fewer cells in the overlap region (Auto), which the
	/// interface tests above rely on. When both grids use the same cell size, the enumeration itself is
	/// symmetric (the straddle window is a cell-unit condition that does not care which cell is mapped
	/// into the other grid), so forcing either side must produce exactly the same contacts. This guards
	/// the role handling of the refactor: contact points and sub shape IDs have to stay assigned to
	/// shape1/shape2 regardless of which side iterates.
	TEST_CASE("VoxelShapeIterateSideEquivalence")
	{
		// ★ 固定用 mode 3：它逐对取"格心连线"，两个方向严格镜像 ⇒ 总深度必须逐位相同。
		//   mode 4 的 2x2x2 邻域是**以候选侧那一格为基准**张开的（用户给的算法定义如此），
		//   交换遍历方向候选侧就换人 ⇒ 邻域不同、总深度可以不同（实测 19.06 vs 20.00）。
		//   这是算法本身的性质，不是回归，所以这条对称性用例不覆盖 mode 4。
		NormalModeScope normal_mode_scope(3);

		// A wide 2.5 x 2.5 m slab and a 0.625 m cube, both using cVoxelSize cells
		VoxelObject slab(40, 4, 40, cVoxelSize);
		sMakeCube(slab);
		VoxelObject cube(10, 10, 10, cVoxelSize);
		sMakeCube(cube);

		CollideShapeSettings settings;
		settings.mMaxSeparationDistance = 0.02f;

		struct Result
		{
			int				mNumContacts;
			float			mTotalDepth;
			Vec3			mPointSum;
			Vec3			mFirstPoint1;
			Vec3			mFirstPoint2;
			bool			mHasFirst = false;
		};

		// The cube is stacked on the slab with an overlap of 0.05 m (which makes it 4.6 contacts per
		// cell of the contact plane) and offset by a voxel so that no plane is exactly aligned.
		Mat44 t1 = Mat44::sTranslation(Vec3(0, 0, 0));
		Mat44 t2 = Mat44::sTranslation(Vec3(0.3f * cVoxelSize, 7.0f * cVoxelSize - 0.05f, -0.4f * cVoxelSize));

		auto query = [&](int inSide)
		{
			IterateSideScope scope(inSide);

			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(slab.mShape, t1, cube.mShape, t2, settings, collector);

			Result r;
			r.mNumContacts = int(collector.mHits.size());
			r.mTotalDepth = 0.0f;
			r.mPointSum = Vec3::sZero();
			for (const CollideShapeResult &h : collector.mHits)
			{
				r.mTotalDepth += max(0.0f, h.mPenetrationDepth);
				r.mPointSum += h.mContactPointOn1 + h.mContactPointOn2;
			}
			if (!collector.mHits.empty())
			{
				r.mFirstPoint1 = collector.mHits[0].mContactPointOn1;
				r.mFirstPoint2 = collector.mHits[0].mContactPointOn2;
				r.mHasFirst = true;
			}
			return r;
		};

		Result side1 = query(1);
		Result side2 = query(2);

		Trace("iterate side 1: %d contacts, total depth %.4f", side1.mNumContacts, side1.mTotalDepth);
		Trace("iterate side 2: %d contacts, total depth %.4f", side2.mNumContacts, side2.mTotalDepth);

		CHECK(side1.mNumContacts > 0);
		CHECK(side1.mNumContacts == side2.mNumContacts);
		CHECK_APPROX_EQUAL(side1.mTotalDepth, side2.mTotalDepth, 1.0e-4f);

		// The contact points have to stay on the surface of shape1 / shape2 no matter which side iterates
		// (they must not swap or mirror), so the sum of both endpoints must match exactly
		CHECK(side1.mPointSum.IsClose(side2.mPointSum, 1.0e-6f));
	}

		/// A voxel block must rest on a voxel floor no matter which side the narrow phase iterates.
	///
	/// The iterate side is picked automatically (the side with fewer cells in the overlap region), which for
	/// a "big grid vs small block" pair is the small block. This used to be broken: the contact points were
	/// derived from the iterate / candidate role instead of the shape1 / shape2 role, so when shape 2
	/// iterated, both points were mirrored to the far side of their cells. Jolt derives the contact
	/// separation from those points (the faces are empty, so ManifoldBetweenTwoFaces falls back to using
	/// them as a single point manifold), saw about 2 cells of separation and dropped every contact as being
	/// out of range - the block fell straight through the floor. It presented as "a corner hitting a face
	/// tunnels through, a face hitting a face does not", because the automatic side selection only flips
	/// for some overlap shapes.
	TEST_CASE("VoxelShapeDropOnVoxelFloorAllIterateSides")
	{
		VoxelObject floor(40, 4, 40, cVoxelSize);
		sMakeCube(floor);
		VoxelObject cube(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCube(cube);

		for (float tilt : { 0.0f, 15.0f, 45.0f })
			for (int side : { 0, 1, 2 })			// 0 = automatic, 1 = iterate shape1, 2 = iterate shape2
			{
				IterateSideScope side_scope(side);

				PhysicsTestContext c;
				c.CreateBody(BodyCreationSettings(floor.mShape, RVec3(0, -2.0f * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);
				Quat rotation = Quat::sRotation(Vec3(1, 0, 0), DegreesToRadians(tilt));
				BodyID id = c.GetBodyInterface().CreateAndAddBody(BodyCreationSettings(cube.mShape, RVec3(0.13, 1.5, -0.07), rotation, EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

				c.Simulate(2.0f);

				// The cube has to come to rest on top of the floor (its center ends up at about 0.5 for a flat
				// contact and a bit higher when tilted), not sink into it and certainly not fall through it
				float y = c.GetBodyInterface().GetPosition(id).GetY();
				CHECK(y > 0.3f);
				CHECK(y < 0.9f);

				// And it must have stopped moving vertically
				CHECK(abs(c.GetBodyInterface().GetLinearVelocity(id).GetY()) < 0.05f);
			}

	}

	/// A/B of the interior pair filter (VoxelShape::sSkipInteriorPairs): a cell pair is only skipped when
	/// **both** cells are fully surrounded by solid cells (interior of their own grid). Surface x surface
	/// and surface x solid pairs are unaffected. Both halves of this test use the scenarios that were
	/// reported as broken before (blocks sinking into each other, blocks falling through, blocks getting
	/// jammed), so the filter is only usable if it keeps all of them stable.
	TEST_CASE("VoxelShapeInteriorPairFilter")
	{
		constexpr int cCubeSize = 10;
		constexpr int cFloorSize = 40;

		VoxelObject cube1(cCubeSize, cCubeSize, cCubeSize, cVoxelSize);
		sMakeCube(cube1);
		VoxelObject cube2(cCubeSize, cCubeSize, cCubeSize, cVoxelSize);
		sMakeCube(cube2);
		VoxelObject slab(cFloorSize, 4, cFloorSize, cVoxelSize);
		sMakeCube(slab);

		// ---- 1) Query level: a small cube fully embedded in a big solid block
		//
		// Note: in a face to face stack (a drop or a half voxel offset stack) at least one cell of every
		// pair is a surface cell of its own grid (the boundary of the overlap region always lies on the
		// face of one of the two grids), so those configurations cannot produce interior x interior
		// pairs at all. Only a deeply embedded configuration (a small cube fully inside a big solid
		// block) puts cells that are buried in both grids next to each other - that is what the filter
		// is about.
		{
			VoxelObject big(20, 20, 20, cVoxelSize);		// 2 x 2 x 2 m
			sMakeCube(big);
			VoxelObject small(6, 6, 6, cVoxelSize);			// 0.6 x 0.6 x 0.6 m, fully embedded
			sMakeCube(small);

			CollideShapeSettings settings;
			settings.mMaxSeparationDistance = 0.0f;		// No speculation: exactly one candidate per cell

			// Both grids are centered on the origin, so the cells line up exactly: the straddle
			// enumeration gives exactly one candidate per cell, i.e. one contact for each of the
			// 6 x 6 x 6 = 216 cells of the small cube (well below cMaxVoxelContacts, no truncation).
			Mat44 t1 = Mat44::sTranslation(Vec3(0, 0, 0));
			Mat44 t2 = Mat44::sTranslation(Vec3(0, 0, 0));

			auto query = [&](bool inSkip)
			{
				InteriorPairFilterScope scope(inSkip);
				AllHitCollisionCollector<CollideShapeCollector> collector;
				sCollide(big.mShape, t1, small.mShape, t2, settings, collector);
				return int(collector.mHits.size());
			};

			int without = query(false);
			int with = query(true);

			Trace("interior pair filter: %d contacts without, %d with", without, with);

			CHECK(without == 6 * 6 * 6);
			// The 4 x 4 x 4 = 64 interior cells of the small cube pair up with cells that are buried in
			// the big block as well: the filter has to drop exactly those 64 pairs, while the 152
			// surface cells of the small cube keep generating contacts.
			CHECK(with == without - 4 * 4 * 4);
		}

		// ---- 2) Simulation level: drop, half voxel offset stack and deep interleave must stay stable
		struct SimResult
		{
			double			mY = 0.0;
			float			mDrift = 0.0f;
			float			mUpY = 0.0f;
		};

		// 2a) A 0.625 m cube dropped on a 2.5 x 2.5 m slab
		auto drop = [&](bool inSkip)
		{
			InteriorPairFilterScope scope(inSkip);

			PhysicsTestContext c;
			c.CreateBody(BodyCreationSettings(slab.mShape, RVec3(0, -2.0f * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

			BodyInterface &bi = c.GetBodyInterface();
			BodyID id = bi.CreateAndAddBody(BodyCreationSettings(cube1.mShape, RVec3(0, 0.6, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

			c.Simulate(3.0f);

			RVec3 p = bi.GetPosition(id);
			Quat r = bi.GetRotation(id);
			SimResult res;
			res.mY = p.GetY();
			res.mDrift = Vec3(float(p.GetX()), 0.0f, float(p.GetZ())).Length();
			res.mUpY = (r * Vec3::sAxisY()).GetY();
			return res;
		};

		// 2b) Cube 2 spawned a full voxel deep into cube 1 and half a voxel off in X (the jam report)
		auto interleave = [&](bool inSkip)
		{
			InteriorPairFilterScope scope(inSkip);

			PhysicsTestContext c;
			c.CreateBody(BodyCreationSettings(slab.mShape, RVec3(0, -2.0f * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

			BodyInterface &bi = c.GetBodyInterface();
			BodyID id1 = bi.CreateAndAddBody(BodyCreationSettings(cube1.mShape, RVec3(0, 0.5f * cCubeSize * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);
			BodyID id2 = bi.CreateAndAddBody(BodyCreationSettings(cube2.mShape, RVec3(0.5f * cVoxelSize, 1.5f * cCubeSize * cVoxelSize - cVoxelSize, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

			c.Simulate(3.0f);

			RVec3 p1 = bi.GetPosition(id1);
			RVec3 p2 = bi.GetPosition(id2);
			Quat r2 = bi.GetRotation(id2);
			SimResult res;
			res.mY = p2.GetY();
			res.mDrift = Vec3(float(p1.GetX()), 0.0f, float(p1.GetZ())).Length();
			res.mUpY = (r2 * Vec3::sAxisY()).GetY();
			return res;
		};

		SimResult drop_off = drop(false), drop_on = drop(true);
		Trace("drop          : filter off y %.4f drift %.4f up.y %.4f | on y %.4f drift %.4f up.y %.4f", drop_off.mY, drop_off.mDrift, drop_off.mUpY, drop_on.mY, drop_on.mDrift, drop_on.mUpY);
		sCheckApproxEqual(drop_off.mY, 0.5 * cCubeSize * cVoxelSize, 0.06);
		sCheckApproxEqual(drop_on.mY, 0.5 * cCubeSize * cVoxelSize, 0.06);
		CHECK(drop_off.mUpY > 0.999f);
		CHECK(drop_on.mUpY > 0.999f);

		// 2c) A slab dropped half a voxel off the cell grid of the floor (the "blocks sink into each
		// other / the contact plane turns into a wave" report)
		auto half_offset = [&](bool inSkip)
		{
			InteriorPairFilterScope scope(inSkip);

			PhysicsTestContext c;
			c.CreateBody(BodyCreationSettings(slab.mShape, RVec3(0, -2.0f * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

			BodyInterface &bi = c.GetBodyInterface();
			BodyID id = bi.CreateAndAddBody(BodyCreationSettings(cube1.mShape, RVec3(0.5f * cVoxelSize, 0.6, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

			c.Simulate(3.0f);

			RVec3 p = bi.GetPosition(id);
			Quat r = bi.GetRotation(id);
			SimResult res;
			res.mY = p.GetY();
			res.mDrift = Vec3(float(p.GetX()) - 0.5f * cVoxelSize, 0.0f, float(p.GetZ())).Length();
			res.mUpY = (r * Vec3::sAxisY()).GetY();
			return res;
		};

		SimResult half_off = half_offset(false), half_on = half_offset(true);
		Trace("half offset   : filter off y %.4f drift %.4f up.y %.4f | on y %.4f drift %.4f up.y %.4f", half_off.mY, half_off.mDrift, half_off.mUpY, half_on.mY, half_on.mDrift, half_on.mUpY);
		sCheckApproxEqual(half_off.mY, 0.5 * cCubeSize * cVoxelSize, 0.06);
		sCheckApproxEqual(half_on.mY, 0.5 * cCubeSize * cVoxelSize, 0.06);
		CHECK(half_off.mDrift < 0.05f);
		CHECK(half_on.mDrift < 0.05f);
		CHECK(half_off.mUpY > 0.999f);
		CHECK(half_on.mUpY > 0.999f);

		SimResult jam_off = interleave(false), jam_on = interleave(true);
		Trace("interleave    : filter off lower.y %.4f upper.y %.4f | on lower.y %.4f upper.y %.4f", jam_off.mDrift, jam_off.mY, jam_on.mDrift, jam_on.mY);
		// Lower cube stays on the floor, upper cube climbs out of the wedge and rests on top of it
		sCheckApproxEqual(jam_off.mDrift, 0.0, 0.06);
		sCheckApproxEqual(jam_on.mDrift, 0.0, 0.06);
		CHECK(jam_off.mY > 1.5f * cCubeSize * cVoxelSize - 0.06f);
		CHECK(jam_off.mY < 1.5f * cCubeSize * cVoxelSize + 0.06f);
		CHECK(jam_on.mY > 1.5f * cCubeSize * cVoxelSize - 0.06f);
		CHECK(jam_on.mY < 1.5f * cCubeSize * cVoxelSize + 0.06f);
	}

	/// Cost of the voxel vs voxel narrow phase for a resting contact, i.e. the case where the contact
	/// plane is as wide as the grid. This is not a correctness test (no assertions), it only reports the
	/// cost of the query.
	// A collector that only counts hits. AllHitCollisionCollector stores every CollideShapeResult,
	// which is ~800 bytes (two StaticArray<Vec3, 32> faces per result), so its own copying + heap churn
	// dominates the measurement. This one isolates the narrow phase compute (traverse + filter + build
	// the result) from that harness cost, which is exactly what an optimisation of the loop changes.
	class CountingCollideShapeCollector : public CollideShapeCollector
	{
	public:
		virtual void	AddHit(const CollideShapeResult &) override { ++mCount; }
		int				mCount = 0;
	};

	/// Simulation level A/B of the normal rules on the two scenarios the count-only diagnostics single
	/// out, so that "mode 1 wedges / waves" is measured instead of assumed from a contact count:
	/// (a) a cube wedged a full voxel into another one, half a voxel off in X (the historic jam report)
	/// and (b) a slab resting on a floor, half a voxel off in X (the historic "the flat contact plane
	/// turns into a wave" report). Nothing here changes the engine: it only flips
	/// VoxelShape::sNormalMode for the duration of each scene.
	TEST_CASE("VoxelShapeNormalModeSimulation")
	{
		// (a) deep interleave: the upper cube is spawned a full voxel into the lower one, half a voxel
		// off in X, so the bottom layer of the upper cube interleaves with the top layer of the lower one.
		{
			VoxelObject cube1(10, 10, 10, cVoxelSize);
			sMakeCube(cube1);
			VoxelObject cube2(10, 10, 10, cVoxelSize);
			sMakeCube(cube2);
			VoxelObject floor_a(40, 4, 40, cVoxelSize);
			sMakeCube(floor_a);

			for (int mode = 0; mode <= 4; ++mode)
			{
				NormalModeScope s(mode);

				PhysicsTestContext c;
				c.CreateBody(BodyCreationSettings(floor_a.mShape, RVec3(0, -2.0f * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

				BodyInterface &bi = c.GetBodyInterface();
				BodyID id1 = bi.CreateAndAddBody(BodyCreationSettings(cube1.mShape, RVec3(0, 0.5f * 10.0f * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);
				BodyID id2 = bi.CreateAndAddBody(BodyCreationSettings(cube2.mShape, RVec3(0.5f * cVoxelSize, 1.5f * 10.0f * cVoxelSize - cVoxelSize, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

				c.Simulate(3.0f);

				float lower_y = float(bi.GetPosition(id1).GetY());
				float upper_y = float(bi.GetPosition(id2).GetY());
				float up_y = (bi.GetRotation(id2) * Vec3::sAxisY()).GetY();
				Trace("sim interleave mode %d: lower y %.4f upper y %.4f up.y %.4f", mode, lower_y, upper_y, up_y);

				sCheckApproxEqual(lower_y, 0.5f * 10.0f * cVoxelSize, 0.06);
				CHECK(up_y > 0.999f);
				if (mode != 1)		// mode 1 is the documented defect here: it sinks a voxel instead of climbing out
				{
					CHECK(upper_y > 1.5f * 10.0f * cVoxelSize - 0.06f);
					CHECK(upper_y < 1.5f * 10.0f * cVoxelSize + 0.06f);
				}
			}
		}

		// (b) half cell offset stack: a slab dropped on the floor with its cell grid offset half a voxel
		// in X, so the flat contact is between misaligned cell centers.
		{
			VoxelObject bottom(40, 4, 40, cVoxelSize);
			sMakeCube(bottom);
			VoxelObject top(30, 4, 30, cVoxelSize);
			sMakeCube(top);

			for (int mode = 0; mode <= 4; ++mode)
			{
				NormalModeScope s(mode);

				PhysicsTestContext c;
				c.CreateBody(BodyCreationSettings(bottom.mShape, RVec3(0, -2.0f * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

				BodyInterface &bi = c.GetBodyInterface();
				BodyID id = bi.CreateAndAddBody(BodyCreationSettings(top.mShape, RVec3(0.5f * cVoxelSize, 0.3, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

				c.Simulate(3.0f);

				RVec3 p = bi.GetPosition(id);
				float y = float(p.GetY());
				float up_y = (bi.GetRotation(id) * Vec3::sAxisY()).GetY();
				float drift = Vec3(float(p.GetX()) - 0.5f * cVoxelSize, 0.0f, float(p.GetZ())).Length();
				Trace("sim half stack mode %d: y %.4f drift %.4f up.y %.4f", mode, y, drift, up_y);

				sCheckApproxEqual(y, 0.5f * 4.0f * cVoxelSize, 0.04);
				CHECK(drift < 0.05f);
				CHECK(up_y > 0.999f);
			}
		}

		// (c) thin plate: a 1 voxel thick floor (0.1 m). Every cell of such a plate has two exposed faces
		// (top and bottom), so under the fusion rule (>= 2 exposed -> the center line) the plate counts as
		// a ring of edge cells and the contact degenerates to mode 1: this is the documented limitation of
		// the "exposed face count" criterion. With sEdgeEitherSide (the criterion looking at both cells)
		// this happens no matter which side is iterated, so the plate sinks by 1.34 cm here too; with the
		// criterion looking at the iterated side only it depends on which side was picked (it does not,
		// here: the compact cube is iterated, so the fusion used to stay flush at 0.5).
		// A half voxel offset in X makes every cell center line diagonal, like the half offset stack.
		{
			VoxelObject plate(40, 1, 40, cVoxelSize);
			sMakeCube(plate);
			VoxelObject cube(10, 10, 10, cVoxelSize);
			sMakeCube(cube);

			for (int mode = 0; mode <= 4; ++mode)
			{
				NormalModeScope s(mode);

				PhysicsTestContext c;
				c.CreateBody(BodyCreationSettings(plate.mShape, RVec3(0, -0.5f * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

				BodyInterface &bi = c.GetBodyInterface();
				BodyID id = bi.CreateAndAddBody(BodyCreationSettings(cube.mShape, RVec3(0.5f * cVoxelSize, 0.6, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

				c.Simulate(3.0f);

				RVec3 p = bi.GetPosition(id);
				float up_y = (bi.GetRotation(id) * Vec3::sAxisY()).GetY();
				float drift = Vec3(float(p.GetX()) - 0.5f * cVoxelSize, 0.0f, float(p.GetZ())).Length();
				Trace("sim thin plate mode %d: y %.4f drift %.4f up.y %.4f", mode, p.GetY(), drift, up_y);
			}
		}

		// (d) thin plate as the iterated side: here the SMALL object is the 1 voxel thick plate (10 x 10
		// cells vs the floor's 40 x 40), so the narrow phase iterates the plate and the fusion judges the
		// plate's
		// own cells - every one of which has two exposed faces - so the whole contact degenerates to
		// mode 1. This is where the "exposed face count" criterion is weakest, and with sEdgeEitherSide
		// it is also what happens in (c) - a 1 thick plate is never treated as a flat surface.
		{
			VoxelObject floor_d(40, 4, 40, cVoxelSize);
			sMakeCube(floor_d);
			VoxelObject plate(10, 1, 10, cVoxelSize);
			sMakeCube(plate);

			for (int mode = 0; mode <= 4; ++mode)
			{
				NormalModeScope s(mode);

				PhysicsTestContext c;
				c.CreateBody(BodyCreationSettings(floor_d.mShape, RVec3(0, -2.0f * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

				BodyInterface &bi = c.GetBodyInterface();
				BodyID id = bi.CreateAndAddBody(BodyCreationSettings(plate.mShape, RVec3(0.5f * cVoxelSize, 0.3, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

				c.Simulate(3.0f);

				RVec3 p = bi.GetPosition(id);
				float y = float(p.GetY());
				float up_y = (bi.GetRotation(id) * Vec3::sAxisY()).GetY();
				float drift = Vec3(float(p.GetX()) - 0.5f * cVoxelSize, 0.0f, float(p.GetZ())).Length();
				Trace("sim thin iterated mode %d: y %.4f drift %.4f up.y %.4f", mode, y, drift, up_y);

			// One cell thick plate, so the contact geometry is dominated by the cell radius rather than by
			// the normal rule: at this cell size all four rules land on the same resting height and the
			// mode 1 / fusion sink that the 0.1 m cell case showed is no longer separable. What must still
			// hold is that the plate rests on the floor instead of sinking through it.
				CHECK(y > 0.0f);
				CHECK(y < 0.5f * cVoxelSize);
			}
		}
	}

/// Cost A/B of the normal rules. All four replace the whole grid SAT axis, so the only difference is how
/// the per pair direction is derived: 0 = global axis, 1 = per pair center line, 2 = exposed face normal,
/// 3 = the fusion (edge cells -> 1, flat / embedded cells -> 0, i.e. the global axis it already holds).
///
/// The fusion only asks "does this cell have >= 2 exposed faces?". Non edge cells (a flat surface cell or
/// a fully embedded cell) keep the global axis, so they pay the lookup chain and *no* per pair dot at all;
/// only edge cells pay one Length + one divide. The count stops at two, and when the first five neighbours
/// are all solid the sixth is not probed (it would only make the count one, still not an edge).
/// Counting the faces inside the mode 0 / mode 2 branches instead ("probe both faces of every axis" = 6
/// lookups) looks cheaper but measured *slower* (1.19 / 1.27 vs 1.105 / 1.145 relative to mode 2): the
/// separate chain keeps the loop straight line with constant probe directions, which the compiler folds.
///
/// Count only (the collector stores nothing): storing the results is ~800 bytes per contact and
/// dominates the query. Each size is measured with the modes alternating over several rounds - a
/// machine drift of +-40% between runs is normal, alternating cancels it - and the median is printed.
/// The grids are kept small enough not to hit cMaxVoxelContacts, otherwise all modes would stop at
/// the same contact and the per pair cost would be hidden.
TEST_CASE("VoxelShapeNormalModeCost")
{
	constexpr int cNumQueries = 500;
	enum { cRounds = 7, cModes = 4 };

	auto measure = [&](const Shape *inShape, int inMode, float inExtent, int &outContacts)
	{
		PhysicsTestContext c;
		c.CreateBody(BodyCreationSettings(inShape, RVec3::sZero(), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

		CollideShapeSettings settings;
		settings.mMaxSeparationDistance = 0.02f;
		settings.mCollectFacesMode = ECollectFacesMode::NoFaces;

		// Slightly tilted and penetrating by 2%, like the other benchmarks
		RMat44 query_transform = RMat44::sRotationTranslation(Quat::sRotation(Vec3(1, 1, 0.5f).Normalized(), DegreesToRadians(10.0f)), RVec3(0, 0.98f * inExtent, 0));

		NormalModeScope scope(inMode);

		// Warm up
		{
			CountingCollideShapeCollector warm;
			c.GetSystem()->GetNarrowPhaseQuery().CollideShape(inShape, Vec3::sOne(), query_transform, settings, RVec3::sZero(), warm);
			outContacts = warm.mCount;
		}

		chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
		for (int i = 0; i < cNumQueries; ++i)
		{
			CountingCollideShapeCollector counting;
			c.GetSystem()->GetNarrowPhaseQuery().CollideShape(inShape, Vec3::sOne(), query_transform, settings, RVec3::sZero(), counting);
		}
		chrono::microseconds d = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
		return double(d.count()) / cNumQueries;
	};

	auto median = [](const double *inValues)
	{
		double s[cRounds];
		for (int i = 0; i < cRounds; ++i)
			s[i] = inValues[i];
		for (int i = 1; i < cRounds; ++i)
		{
			double k = s[i];
			int j = i - 1;
			while (j >= 0 && s[j] > k) { s[j + 1] = s[j]; --j; }
			s[j + 1] = k;
		}
		return s[cRounds / 2];
	};

	for (int size : { 10, 20 })
	{
		VoxelObject object(size, size, size, cVoxelSize);
		sMakeCube(object);

		double us[cModes][cRounds];
		int contacts[cModes] = { 0, 0, 0, 0 };
		for (int round = 0; round < cRounds; ++round)
			for (int m = 0; m < cModes; ++m)
				us[m][round] = measure(object.mShape, m, size * cVoxelSize, contacts[m]);

		double med[cModes];
		for (int m = 0; m < cModes; ++m)
			med[m] = median(us[m]);

		Trace("normal cost %2d^3 (contacts m0 %d m1 %d m2 %d m3 %d): m0 %.3f | m1 %.3f | m2 %.3f | m3(fusion) %.3f us",
			size, contacts[0], contacts[1], contacts[2], contacts[3], med[0], med[1], med[2], med[3]);
		Trace("    ratio to m2: m0 %.3f | m1 %.3f | m3(fusion) %.3f",
			med[2] > 0.0? med[0] / med[2] : 0.0, med[2] > 0.0? med[1] / med[2] : 0.0, med[2] > 0.0? med[3] / med[2] : 0.0);

		for (int m = 0; m < cModes; ++m)
			CHECK(contacts[m] > 0);
	}
}

	/// Force the edge face count cache (see VoxelShape::sEdgeCacheEnabled) for the duration of a scope
	struct EdgeCacheScope
	{
		EdgeCacheScope(int inEnabled) :
			mOld(VoxelShape::sEdgeCacheEnabled)
		{
			VoxelShape::sEdgeCacheEnabled = inEnabled;
		}

		~EdgeCacheScope()
		{
			VoxelShape::sEdgeCacheEnabled = mOld;
		}

		int				mOld;
	};

	/// Force the fusion criterion to look at one side or both (see VoxelShape::sEdgeEitherSide)
	struct EdgeRuleScope
	{
		EdgeRuleScope(int inEitherSide) :
			mOld(VoxelShape::sEdgeEitherSide)
		{
			VoxelShape::sEdgeEitherSide = inEitherSide;
		}

		~EdgeRuleScope()
		{
			VoxelShape::sEdgeEitherSide = mOld;
		}

		int				mOld;
	};

	/// Are two hit lists the same contact for contact, bit for bit?
	static bool sSameHits(const AllHitCollisionCollector<CollideShapeCollector> &inA, const AllHitCollisionCollector<CollideShapeCollector> &inB)
	{
		if (inA.mHits.size() != inB.mHits.size())
			return false;
		for (size_t i = 0; i < inA.mHits.size(); ++i)
		{
			const CollideShapeResult &a = inA.mHits[i];
			const CollideShapeResult &b = inB.mHits[i];
			if (a.mPenetrationDepth != b.mPenetrationDepth
				|| a.mSubShapeID1 != b.mSubShapeID1
				|| a.mSubShapeID2 != b.mSubShapeID2
				|| a.mContactPointOn1.GetX() != b.mContactPointOn1.GetX()
				|| a.mContactPointOn1.GetY() != b.mContactPointOn1.GetY()
				|| a.mContactPointOn1.GetZ() != b.mContactPointOn1.GetZ()
				|| a.mContactPointOn2.GetX() != b.mContactPointOn2.GetX()
				|| a.mContactPointOn2.GetY() != b.mContactPointOn2.GetY()
				|| a.mContactPointOn2.GetZ() != b.mContactPointOn2.GetZ()
				|| a.mPenetrationAxis.GetX() != b.mPenetrationAxis.GetX()
				|| a.mPenetrationAxis.GetY() != b.mPenetrationAxis.GetY()
				|| a.mPenetrationAxis.GetZ() != b.mPenetrationAxis.GetZ())
				return false;
		}
		return true;
	}

/// The edge face count cache (VoxelShape::sEdgeCacheEnabled) must be invisible. It stores an answer that
/// only depends on the voxel data of one cell, so switching it on must not move a single contact - and the
/// second half of the test is the one that keeps an editable world correct: after the voxel data changed,
/// Recalculate() hands out a new voxel data version, which is part of the cache key, so stale entries can
/// never be used.
TEST_CASE("VoxelShapeEdgeCacheEquivalence")
{
	// 这个用例只测 mode 3 的"数外露面"缓存（mode 4 根本不查它），所以固定用 mode 3。
	NormalModeScope normal_mode_scope(3);

	constexpr int cSize = 20;

	VoxelObject object(cSize, cSize, cSize, cVoxelSize);
	sMakeCube(object);
	const VoxelShape *shape = static_cast<const VoxelShape *>(object.mShape.GetPtr());

	// Same arrangement as the cost benchmarks: tilted and penetrating by 2%
	Mat44 transform = Mat44::sRotationTranslation(Quat::sRotation(Vec3(1, 1, 0.5f).Normalized(), DegreesToRadians(10.0f)), Vec3(0, 0.98f * cSize * cVoxelSize, 0));

	CollideShapeSettings settings;
	settings.mMaxSeparationDistance = 0.02f;
	settings.mCollectFacesMode = ECollectFacesMode::NoFaces;

	auto collide = [&](AllHitCollisionCollector<CollideShapeCollector> &outCollector)
	{
		sCollide(shape, transform, shape, Mat44::sIdentity(), settings, outCollector);
	};

	// 1) cache off vs cache on, twice: the second run is the one that actually reads the cache
	AllHitCollisionCollector<CollideShapeCollector> raw, cached_first, cached_second;
	{
		EdgeCacheScope raw_scope(0);
		collide(raw);
	}
	{
		EdgeCacheScope cached_scope(1);
		collide(cached_first);

		VoxelShape::sEdgeCacheStats = 1;
		VoxelShape::sEdgeCacheHits = 0;
		VoxelShape::sEdgeCacheMisses = 0;
		collide(cached_second);
		uint64 hits = VoxelShape::sEdgeCacheHits, misses = VoxelShape::sEdgeCacheMisses;
		VoxelShape::sEdgeCacheStats = 0;

		Trace("edge cache equivalence: %llu hits / %llu lookups on a repeated query", (unsigned long long)hits, (unsigned long long)(hits + misses));

		// The cache must actually be doing something, otherwise the comparison below proves nothing
		CHECK(hits > 0);
	}

	CHECK(raw.mHits.size() > 0);
	CHECK(sSameHits(raw, cached_first));
	CHECK(sSameHits(raw, cached_second));

	// 2) change the voxel data and ask again - the cached answers must not leak into the new data. The edit
	// is picked so that it *has* to change the classification of the contacted cells: the two cubes touch
	// with their bottom and top layer (y = 0 and y = size - 1), and clearing the layer right next to them
	// (y = 1 and y = size - 2, in an x band so that contacts survive) gives those two layers a second
	// exposed face - they turn from "flat surface" cells into edge cells, i.e. from the global axis into
	// the per pair normal. A stale cache would keep the old classification and produce different hits.
	for (int z = 0; z < cSize; ++z)
		for (int x = 4; x <= 15; ++x)
		{
			object.Cell(x, 1, z) = 0;
			object.Cell(x, cSize - 2, z) = 0;
		}
	object.Update();

	AllHitCollisionCollector<CollideShapeCollector> after_raw, after_cached;
	{
		EdgeCacheScope raw_scope(0);
		collide(after_raw);
	}
	{
		EdgeCacheScope cached_scope(1);
		collide(after_cached);
	}

	CHECK(after_raw.mHits.size() > 0);		// the edit removed contacts but not all of them
	CHECK(sSameHits(after_raw, after_cached));
	CHECK(!sSameHits(raw, after_raw));		// the edit really did change the contacts
}

/// Cost of the edge face count cache (see VoxelShape::sEdgeCacheEnabled), measured with the same
/// count-only / alternating-rounds / median harness as VoxelShapeNormalModeCost so that the numbers can be
/// read next to each other:
///   m0            = the floor: the global axis with no face counting at all
///   m3 raw        = the fusion with the cache off (what the previous commit measured)
///   m3 cached     = the fusion with the cache on, warm (the same query repeated, which is the case the
///                   cache exists for: a sustained contact asks about the same cells every frame)
///   cold raw      = the shape is rebuilt before every query, so Recalculate() hands out a new voxel data
///   cold cached     version and nothing can be in the cache. Recalculate dominates both rows, but it is
///                   identical in both, so the difference is what a miss + insert costs per cell.
///
/// Only sNormalMode == 3 counts exposed faces, so mode 0 ignores the cache entirely - it is the reference
/// for "how much is left to win at all" (the whole count is only ~5% of the query).
TEST_CASE("VoxelShapeEdgeCacheCost")
{
	constexpr int cNumQueries = 500;
	enum { cRounds = 9 };

	auto measure = [&](const VoxelShape *inShape, int inMode, bool inCache, bool inRebuild, float inExtent, int &outContacts)
	{
		PhysicsTestContext c;
		c.CreateBody(BodyCreationSettings(inShape, RVec3::sZero(), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

		CollideShapeSettings settings;
		settings.mMaxSeparationDistance = 0.02f;
		settings.mCollectFacesMode = ECollectFacesMode::NoFaces;

		// Slightly tilted and penetrating by 2%, like the other benchmarks
		RMat44 query_transform = RMat44::sRotationTranslation(Quat::sRotation(Vec3(1, 1, 0.5f).Normalized(), DegreesToRadians(10.0f)), RVec3(0, 0.98f * inExtent, 0));

		NormalModeScope mode_scope(inMode);
		EdgeCacheScope cache_scope(inCache? 1 : 0);

		{
			CountingCollideShapeCollector warm;
			c.GetSystem()->GetNarrowPhaseQuery().CollideShape(inShape, Vec3::sOne(), query_transform, settings, RVec3::sZero(), warm);
			outContacts = warm.mCount;
		}

		chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
		for (int i = 0; i < cNumQueries; ++i)
		{
			if (inRebuild)
				inShape->Recalculate();
			CountingCollideShapeCollector counting;
			c.GetSystem()->GetNarrowPhaseQuery().CollideShape(inShape, Vec3::sOne(), query_transform, settings, RVec3::sZero(), counting);
		}
		chrono::microseconds d = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
		return double(d.count()) / cNumQueries;
	};

	auto median = [](const double *inValues)
	{
		double s[cRounds];
		for (int i = 0; i < cRounds; ++i)
			s[i] = inValues[i];
		for (int i = 1; i < cRounds; ++i)
		{
			double k = s[i];
			int j = i - 1;
			while (j >= 0 && s[j] > k) { s[j + 1] = s[j]; --j; }
			s[j + 1] = k;
		}
		return s[cRounds / 2];
	};

	/// The fastest round of the set. On a machine that is not idle the median carries whatever else was
	/// running; noise only ever *adds* time, so the minimum is the stable estimate of the real cost - and
	/// this comparison is a 3% question, so it has to be stable.
	auto best = [](const double *inValues)
	{
		double m = inValues[0];
		for (int i = 1; i < cRounds; ++i)
			if (inValues[i] < m)
				m = inValues[i];
		return m;
	};

	for (int size : { 10, 20 })
	{
		VoxelObject object(size, size, size, cVoxelSize);
		sMakeCube(object);
		const VoxelShape *shape = static_cast<const VoxelShape *>(object.mShape.GetPtr());
		float extent = size * cVoxelSize;

		double m0[cRounds], raw[cRounds], cached[cRounds], cold_raw[cRounds], cold_cached[cRounds];
		int contacts = 0;
		for (int round = 0; round < cRounds; ++round)
		{
			m0[round] = measure(shape, 0, false, false, extent, contacts);
			// raw and cached are the comparison that matters, so their order is flipped every other round:
			// a fixed order would let a monotonic drift (warming up / cooling down) leak into the ratio
			if ((round & 1) == 0)
			{
				raw[round] = measure(shape, 3, false, false, extent, contacts);
				cached[round] = measure(shape, 3, true, false, extent, contacts);
			}
			else
			{
				cached[round] = measure(shape, 3, true, false, extent, contacts);
				raw[round] = measure(shape, 3, false, false, extent, contacts);
			}
			cold_raw[round] = measure(shape, 3, false, true, extent, contacts);
			cold_cached[round] = measure(shape, 3, true, true, extent, contacts);
		}

		Trace("edge cache cost %2d^3 (%d contacts): m0 %.3f | m3 raw %.3f | m3 cached %.3f us",
			size, contacts, median(m0), median(raw), median(cached));
		Trace("    vs m3 raw: m0 %.3f | m3 cached %.3f ; cold (rebuild per query, Recalculate included) raw %.3f cached %.3f",
			median(m0) / median(raw), median(cached) / median(raw), median(cold_raw), median(cold_cached));
		Trace("    best of %d rounds: m0 %.3f | m3 raw %.3f | m3 cached %.3f us -> m0/raw %.3f | cached/raw %.3f",
			cRounds, best(m0), best(raw), best(cached), best(m0) / best(raw), best(cached) / best(raw));

		// Hit rate of the warm case. The cache is process wide and keyed by the voxel data version, which
		// does not change here, so by now it is already warm: this is the steady state of a sustained contact.
		{
			VoxelShape::sEdgeCacheStats = 1;
			VoxelShape::sEdgeCacheHits = 0;
			VoxelShape::sEdgeCacheMisses = 0;
			measure(shape, 3, true, false, extent, contacts);
			uint64 hits = VoxelShape::sEdgeCacheHits, total = hits + VoxelShape::sEdgeCacheMisses;
			VoxelShape::sEdgeCacheStats = 0;
			Trace("    edge cache warm hit rate %.2f%% (%llu hits / %llu lookups)", total > 0? 100.0 * double(hits) / double(total) : 0.0, (unsigned long long)hits, (unsigned long long)total);
		}

		CHECK(contacts > 0);
	}
}

/// What the fusion criterion costs when it looks at BOTH sides (VoxelShape::sEdgeEitherSide == 1) instead
/// of only the iterated side (== 0).
///
/// The extra work only shows up when the iterated cell is NOT an edge cell: then the other cell has to be
/// counted as well, and that one is different for every cell pair (the iterated side is counted once per
/// cell and reused by all of its candidates).
///
/// The query shape is the same voxel grid transformed, so the two lattices are rotated relative to each
/// other. Two tilts: 10 degrees (the same as the other benchmarks; the contacted cells are mostly flat
/// surface cells, so this is the common flat-on-flat case) and 45 degrees (the surviving cell pairs are
/// much more off axis, so the two rules disagree far more often).
///
/// Alternating exes is not possible here (both variants live in the same exe), so instead the variant order
/// is rotated every round and the reported number is the minimum over the rounds - see the note on the
/// measurement discipline in VoxelShapeEdgeCacheCost.
TEST_CASE("VoxelShapeEdgeRuleCost")
{
	constexpr int cNumQueries = 500;
	enum { cRounds = 9, cVariants = 4 };

	auto measure = [&](const VoxelShape *inShape, int inVariant, float inExtent, int inTiltDegrees, int &outContacts)
	{
		PhysicsTestContext c;
		c.CreateBody(BodyCreationSettings(inShape, RVec3::sZero(), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

		CollideShapeSettings settings;
		settings.mMaxSeparationDistance = 0.02f;
		settings.mCollectFacesMode = ECollectFacesMode::NoFaces;

		RMat44 query_transform = RMat44::sRotationTranslation(Quat::sRotation(Vec3(1, 1, 0.5f).Normalized(), DegreesToRadians(float(inTiltDegrees))), RVec3(0, 0.98f * inExtent, 0));

		// 0 = mode 0, 1 = fusion judging the iterated side only, 2 = fusion judging both sides, 3 = mode 2
		int mode = inVariant == 0? 0 : inVariant == 3? 2 : 3;
		NormalModeScope mode_scope(mode);
		EdgeCacheScope cache_scope(1);						// both fusion variants run with the cache on
		EdgeRuleScope rule_scope(inVariant == 1? 0 : 1);	// the criterion only exists in mode 3

		{
			CountingCollideShapeCollector warm;
			c.GetSystem()->GetNarrowPhaseQuery().CollideShape(inShape, Vec3::sOne(), query_transform, settings, RVec3::sZero(), warm);
			outContacts = warm.mCount;
		}

		chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
		for (int i = 0; i < cNumQueries; ++i)
		{
			CountingCollideShapeCollector counting;
			c.GetSystem()->GetNarrowPhaseQuery().CollideShape(inShape, Vec3::sOne(), query_transform, settings, RVec3::sZero(), counting);
		}
		chrono::microseconds d = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
		return double(d.count()) / cNumQueries;
	};

	auto best = [](const double *inValues)
	{
		double m = inValues[0];
		for (int i = 1; i < cRounds; ++i)
			if (inValues[i] < m)
				m = inValues[i];
		return m;
	};

	for (int tilt : { 10, 45 })
		for (int size : { 10, 20 })
		{
			VoxelObject object(size, size, size, cVoxelSize);
			sMakeCube(object);
			const VoxelShape *shape = static_cast<const VoxelShape *>(object.mShape.GetPtr());
			float extent = size * cVoxelSize;

			double t[cVariants][cRounds];
			int contacts[cVariants] = { 0, 0, 0, 0 };
			for (int round = 0; round < cRounds; ++round)
				for (int k = 0; k < cVariants; ++k)
				{
					// rotate the order every round so a monotonic drift cannot bias one variant
					int v = (round + k) % cVariants;
					t[v][round] = measure(shape, v, extent, tilt, contacts[v]);
				}

			double m0 = best(t[0]), owner = best(t[1]), both = best(t[2]), m2 = best(t[3]);
			Trace("edge rule cost %2d deg %2d^3 (contacts m0 %d | m2 %d | fusion owner-only %d | both %d): m0 %.3f | m2 %.3f | m3 owner-only %.3f | m3 both %.3f us",
				tilt, size, contacts[0], contacts[3], contacts[1], contacts[2], m0, m2, owner, both);
			Trace("    vs m2: m0 %.3f | m3 owner-only %.3f | m3 both %.3f ; both / owner-only %.3f",
				m2 > 0.0? m0 / m2 : 0.0, m2 > 0.0? owner / m2 : 0.0, m2 > 0.0? both / m2 : 0.0, owner > 0.0? both / owner : 0.0);

			CHECK(contacts[1] == contacts[2]);
		}
}

/// The primitives the normal rules are built from, measured in isolation, so that the cost difference
/// reported by VoxelShapeNormalModeCost can be attributed instead of guessed.
///
/// Per cell pair the rules do:
///   mode 2 (exposed face normal):      <= 3 voxel lookups + 3 dot products
///   mode 1 (per pair center line):        0 voxel lookups + 1 Length + 1 divide
///   mode 3 (fusion):                  1~6 voxel lookups (the exposed face count, early out at two)
///                                     + then either mode 1 (edge cell) or nothing at all (non edge,
///                                     the global axis is already in place)
///
/// This measures the two primitives that make up the difference: a voxel lookup (6 range checks +
/// index arithmetic + chunk occupancy mask) and a normalize (one dot, one sqrt, one divide and three
/// multiplies). It is the input for predicting what the fusion costs before running the benchmark.
TEST_CASE("VoxelShapeNormalPrimitiveCost")
{
	enum { cGridSize = 32, cIters = 1 << 22 };

	VoxelObject object(cGridSize, cGridSize, cGridSize, cVoxelSize);
	sMakeCube(object);
	const VoxelShape *shape = static_cast<const VoxelShape *>(object.mShape.GetPtr());

	volatile int sink = 0;

	// N voxel lookups per iteration. The index walks the grid (data dependent) and the voxel data is
	// only known at run time, so nothing can be folded away.
	auto measure_lookups = [&](int inCount, const char *inName)
	{
		int acc = 0;
		chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
		for (int i = 0; i < cIters; ++i)
		{
			int x = (i * 7) & (cGridSize - 1), y = (i * 13) & (cGridSize - 1), z = (i * 29) & (cGridSize - 1);
			for (int k = 0; k < inCount; ++k)
				acc += shape->IsSolid((x + k) & (cGridSize - 1), y, z)? 1 : 0;
		}
		chrono::microseconds d = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
		sink = acc;
		Trace("%s: %7.3f ns per cell pair", inName, double(d.count()) * 1000.0 / cIters);
	};
	measure_lookups(1, "1 lookup ");
	measure_lookups(2, "2 lookups");
	measure_lookups(3, "3 lookups");
	measure_lookups(6, "6 lookups");

	// The normalize that mode 1 (and the fusion edge branch) pays and mode 2 does not: one dot, one
	// sqrt, one divide and three multiplies. Same measurement trick as above: run the loop twice and
	// take the difference, which keeps the operand build cost out of the number.
	volatile float vk = 1.0f;
	volatile float sink_f = 0.0f;
	auto run_normalize_loop = [&](bool inNormalize)
	{
		float acc = 0.0f;
		chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
		if (inNormalize)
			for (int i = 0; i < cIters; ++i)
			{
				Vec3 v(float(i & 7) * vk + 1.0f, float((i >> 3) & 7) * vk + 2.0f, float((i >> 6) & 7) * vk + 3.0f);
				acc += v.Normalized().GetX();
			}
		else
			for (int i = 0; i < cIters; ++i)
			{
				Vec3 v(float(i & 7) * vk + 1.0f, float((i >> 3) & 7) * vk + 2.0f, float((i >> 6) & 7) * vk + 3.0f);
				acc += v.GetX();
			}
		chrono::microseconds d = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);
		sink_f = acc;
		return double(d.count()) * 1000.0 / cIters;
	};
	double ns_build = run_normalize_loop(false);
	double ns_norm = run_normalize_loop(true) - ns_build;
	Trace("1 normalize (mode 1 / fusion edge):%7.3f ns per cell pair (operand build %.3f)", ns_norm, ns_build);

	CHECK(sink >= 0);
}

	/// Quantifies the "voxel blocks float" report.
	///
	/// A 10^3 cube (1 m) resting on a 40 x 4 x 40 voxel floor, once per normal mode. The floor top face is
	/// at y = 0, so a cube resting flush has its center at exactly y = 0.5. The voxel cells use the
	/// inscribed sphere (radius 0.5 cell, see sGetUnitSphere) whose top is exactly on the cell face, so a
	/// flush contact gives depth 0 - a gap can only come from the contact normal:
	///     depth = (r1 + r2) - |delta_center . normal|
	/// grows as soon as the normal is not parallel to the cell center line, and the solver pushes the cube
	/// out along that normal. The same drop is also done on a BoxShape floor, which goes through the
	/// voxel vs convex path (the cell spheres), so that one isolates "did the sphere radius change".
	TEST_CASE("VoxelShapeNormalModeRest")
	{
		VoxelObject floor(40, 4, 40, cVoxelSize);
		sMakeCube(floor);
		VoxelObject cube(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCube(cube);

		// 0~4 静止时都不许被顶起来（y 必须留在 0.4 以上）。
		for (int mode = 0; mode <= 4; ++mode)
		{
			{
				NormalModeScope scope(mode);

				PhysicsTestContext c;
				c.CreateBody(BodyCreationSettings(floor.mShape, RVec3(0, -2.0f * cVoxelSize, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);
				BodyID id = c.GetBodyInterface().CreateAndAddBody(BodyCreationSettings(cube.mShape, RVec3(0, 1.5, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

				c.Simulate(3.0f);

				RVec3 p = c.GetBodyInterface().GetPosition(id);
				Quat r = c.GetBodyInterface().GetRotation(id);
				Trace("rest voxel floor, mode %d: y %.4f (flush 0.5, gap %+.4f) up.y %.4f",
					mode, p.GetY(), p.GetY() - 0.5, (r * Vec3::sAxisY()).GetY());
				CHECK(p.GetY() > 0.4f);
				CHECK(p.GetY() < 1.2f);

				// It must not float. A flush contact rests at 0.5 minus the solver slop; anything above
				// 0.505 means the contact normal is not parallel to the cell center line and the depth
				// (r1 + r2 - |delta_center . normal|) is inflated. This is the "voxel blocks float"
				// symptom: the refuted field gradient normal did exactly that (+0.02 m), while 0/1/2 and
				// the fusion stay at 0.48.
				CAPTURE(mode);
				CHECK(p.GetY() < 0.505f);
			}

			{
				NormalModeScope scope(mode);

				PhysicsTestContext c;
				c.CreateFloor();
				BodyID id = c.GetBodyInterface().CreateAndAddBody(BodyCreationSettings(cube.mShape, RVec3(0, 1.5, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

				c.Simulate(3.0f);

				RVec3 p = c.GetBodyInterface().GetPosition(id);
				Trace("rest box floor,   mode %d: y %.4f (flush 0.5, gap %+.4f)", mode, p.GetY(), p.GetY() - 0.5);
				CHECK(p.GetY() > 0.4f);
				CHECK(p.GetY() < 1.2f);
			}
		}
	}

	TEST_CASE("VoxelShapeSphereModePerformance")
	{
		constexpr int cNumQueries = 1000;

		auto measure = [&](const Shape *inShape, const char *inName, float inExtent) {
			PhysicsTestContext c;
			c.CreateBody(BodyCreationSettings(inShape, RVec3::sZero(), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

			CollideShapeSettings settings;

			// The separation that PhysicsSystem uses (mSpeculativeContactDistance), so this measures the
			// same configuration that a running simulation hits
			settings.mMaxSeparationDistance = 0.02f;
			settings.mCollectFacesMode = ECollectFacesMode::CollectFaces;

			// Slightly tilted so that the contact is not perfectly axis aligned, penetrating by 2%
			RMat44 query_transform = RMat44::sRotationTranslation(Quat::sRotation(Vec3(1, 1, 0.5f).Normalized(), DegreesToRadians(10.0f)), RVec3(0, 0.98f * inExtent, 0));

			// Warm up
			int num_hits = 0;
			{
				AllHitCollisionCollector<CollideShapeCollector> collector;
				c.GetSystem()->GetNarrowPhaseQuery().CollideShape(inShape, Vec3::sOne(), query_transform, settings, RVec3::sZero(), collector);
				num_hits = int(collector.mHits.size());

				CountingCollideShapeCollector warmup;
				c.GetSystem()->GetNarrowPhaseQuery().CollideShape(inShape, Vec3::sOne(), query_transform, settings, RVec3::sZero(), warmup);
			}

			chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
			for (int i = 0; i < cNumQueries; ++i)
			{
				AllHitCollisionCollector<CollideShapeCollector> collector;
				c.GetSystem()->GetNarrowPhaseQuery().CollideShape(inShape, Vec3::sOne(), query_transform, settings, RVec3::sZero(), collector);
			}
			chrono::microseconds full = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);

			start = chrono::high_resolution_clock::now();
			for (int i = 0; i < cNumQueries; ++i)
			{
				CountingCollideShapeCollector counting;
				c.GetSystem()->GetNarrowPhaseQuery().CollideShape(inShape, Vec3::sOne(), query_transform, settings, RVec3::sZero(), counting);
			}
			chrono::microseconds count_only = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);

			Trace("%s: %.3f us / query (store all, %d contacts) | %.3f us / query (count only)",
				inName, double(full.count()) / cNumQueries, num_hits, double(count_only.count()) / cNumQueries);
		};

		for (int size : { 10, 20, 32 })
		{
			VoxelObject object(size, size, size, cVoxelSize);
			sMakeCube(object);

			char name[64];
			snprintf(name, sizeof(name), "VoxelShape %2d^3, voxel vs voxel", size);
			measure(object.mShape, name, size * cVoxelSize);
		}
	}

	/// Cost of the primitives that the voxel vs mesh path can use per cell.
	///
	/// The path turns every solid cell into a shape and collides it with the mesh through
	/// CollisionDispatch::sCollideShapeVsShape. This measures two options for a run of N cells along an
	/// axis: N unitsphere calls (one per cell, what the path does today) versus one capsule covering the
	/// whole run. The capsule is more expensive per triangle (Jolt has a dedicated sphere vs mesh path,
	/// while a capsule goes through the generic convex vs mesh one), but it pays the per call overhead
	/// (shape dispatch, mesh traversal setup) once instead of N times - which wins from ~4 cells on:
	///
	///     run  2: capsule is  ~0.95x the spheres
	///     run  4: capsule is  ~0.83x
	///     run  8: capsule is  ~0.73x
	///     run 16: capsule is  ~0.64x
	///
	/// Note that a capsule only allows a uniform scale, so the shape has to be built for the run length
	/// (the cylinder half height carries it) - a pool of capsules indexed by run length, built in cell
	/// units and scaled by the cell size, is what the path would use.
	TEST_CASE("VoxelShapeMeshRunPrimitiveCost")
	{
		constexpr int cNumQueries = 2000;
		const float cell = 0.1f;								// 1 voxel

		// Mesh floor: n x n cells of 2 triangles each, centered on the origin
		const int n = 16;
		TriangleList triangles;
		for (int x = 0; x < n; ++x)
			for (int z = 0; z < n; ++z)
			{
				float x1 = (x - 0.5f * n) * cell, z1 = (z - 0.5f * n) * cell, x2 = x1 + cell, z2 = z1 + cell;
				triangles.push_back(Triangle(Vec3(x1, 0, z1), Vec3(x1, 0, z2), Vec3(x2, 0, z1)));
				triangles.push_back(Triangle(Vec3(x1, 0, z2), Vec3(x2, 0, z2), Vec3(x2, 0, z1)));
			}
		RefConst<Shape> mesh = MeshShapeSettings(triangles).Create().Get();
		Mat44 t_mesh = Mat44::sIdentity();

		CollideShapeSettings settings;
		settings.mMaxSeparationDistance = 0.02f;
		settings.mCollectFacesMode = ECollectFacesMode::CollectFaces;

		// Unit primitives scaled to one cell: radius 0.5 * cell = 0.05 m, the cell center 0.01 m inside the floor
		RefConst<Shape> sphere = new SphereShape(0.5f);
		Vec3 prim_scale = Vec3::sReplicate(cell);
		const float y = 0.4f * cell;

		auto sTime = [&](auto &inCollide)
		{
			CountingCollideShapeCollector warmup;
			inCollide(warmup);

			chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
			for (int i = 0; i < cNumQueries; ++i)
			{
				CountingCollideShapeCollector counting;
				inCollide(counting);
			}
			return double(chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start).count()) / cNumQueries;
		};

		for (int run : { 2, 4, 8, 16 })
		{
			// run spheres: one collide call per cell, as the current voxel vs mesh path does
			auto collide_spheres = [&](CollideShapeCollector &ioCollector)
			{
				for (int i = 0; i < run; ++i)
					CollisionDispatch::sCollideShapeVsShape(sphere, mesh, prim_scale, Vec3::sOne(),
						Mat44::sTranslation(Vec3((i - 0.5f * (run - 1)) * cell, y, 0)),
						t_mesh, SubShapeIDCreator(), SubShapeIDCreator(), settings, ioCollector);
			};

			// One capsule covering the run. Note the shape has to be built for the run length: a capsule
			// only allows a uniform scale, so the cylinder half height carries the run length.
			RefConst<Shape> capsule = new CapsuleShape(0.5f * (run - 1), 0.5f);
			Mat44 m_capsule = Mat44::sRotationTranslation(Quat::sRotation(Vec3::sAxisZ(), DegreesToRadians(-90.0f)), Vec3(0, y, 0));
			auto collide_capsule = [&](CollideShapeCollector &ioCollector)
			{
				CollisionDispatch::sCollideShapeVsShape(capsule, mesh, prim_scale, Vec3::sOne(), m_capsule, t_mesh, SubShapeIDCreator(), SubShapeIDCreator(), settings, ioCollector);
			};

			double t_spheres = sTime(collide_spheres);
			double t_capsule = sTime(collide_capsule);
			Trace("run %2d cells: %2d spheres %.3f us | 1 capsule %.3f us | capsule costs %.2fx the spheres", run, run, t_spheres, t_capsule, t_capsule / t_spheres);
		}
	}

	/// The voxel vs other shape paths (convex shape / mesh / height field / plane), which is what
	/// VoxelShapeRunMerge optimizes: a cube resting on a floor is only touched through its bottom layer,
	/// i.e. size x size cells (100 at 10^3, 1024 at 32^3). Every one of those cells used to be a separate
	/// query; with the run merge only one capsule per row is queried (size of them) and the contact count
	/// drops accordingly. This is the benchmark to watch when touching either path.
	TEST_CASE("VoxelShapeVsOtherShapePerformance")
	{
		constexpr int cNumQueries = 300;

		// Floors whose top face is at y = 0
		RefConst<Shape> box_floor = new BoxShape(Vec3(5, 0.5f, 5));
		Mat44 box_floor_transform = Mat44::sTranslation(Vec3(0, -0.5f, 0));
		RefConst<Shape> mesh_floor = sCreateMeshFloor();
		RefConst<Shape> height_field_floor = sCreateHeightFieldFloor();

		for (int size : { 10, 20, 32 })
		{
			VoxelObject object(size, size, size, cVoxelSize);
			sMakeCube(object);

			// The bottom layer of cells sits 0.01 m inside the floor (the centers of the bottom cells are
			// half a cell above the bottom face of the cube)
			float half_extent = 0.5f * size * cVoxelSize;
			Mat44 transform = Mat44::sTranslation(Vec3(0, half_extent - 0.01f, 0));

			auto measure = [&](const Shape *inOther, Mat44Arg inOtherTransform, ECollectFacesMode inFacesMode, const char *inName)
			{
				CollideShapeSettings settings;
				settings.mMaxSeparationDistance = 0.0f;
				settings.mCollectFacesMode = inFacesMode;

				auto query = [&]() {
					AllHitCollisionCollector<CollideShapeCollector> collector;
					CollisionDispatch::sCollideShapeVsShape(object.mShape, inOther, Vec3::sOne(), Vec3::sOne(), transform, inOtherTransform, SubShapeIDCreator(), SubShapeIDCreator(), settings, collector);
					return int(collector.mHits.size());
				};

				int hits = query();

				chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
				int checksum = 0;
				for (int i = 0; i < cNumQueries; ++i)
					checksum += query();
				chrono::microseconds duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);

				Trace("Voxel %2d^3 vs %-16s %-9s: %8.3f us / query (%d hits)", size, inName,
					inFacesMode == ECollectFacesMode::CollectFaces? "faces" : "no faces",
					double(duration.count()) / cNumQueries, hits);
				CHECK(checksum == hits * cNumQueries);
				CHECK(hits > 0);
			};

			measure(box_floor, box_floor_transform, ECollectFacesMode::CollectFaces, "box floor");
			measure(box_floor, box_floor_transform, ECollectFacesMode::NoFaces, "box floor");
			measure(mesh_floor, Mat44::sIdentity(), ECollectFacesMode::CollectFaces, "mesh floor");
			measure(height_field_floor, Mat44::sIdentity(), ECollectFacesMode::CollectFaces, "height field");
		}
	}

	TEST_CASE("VoxelShapeVsConvexShapes")
	{
		VoxelObject cube(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCube(cube);

		// Overlapping a regular box shape by 0.05
		{
			RefConst<Shape> box = new BoxShape(Vec3(cHalfExtent, cHalfExtent, cHalfExtent));
			CollideShapeSettings settings;
			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(cube.mShape, Mat44::sIdentity(), box, Mat44::sTranslation(Vec3(0, 2.0f * cHalfExtent - 0.05f, 0)), settings, collector);
			CHECK(collector.mHits.size() > 0);
			if (!collector.mHits.empty())
			{
				// The box is a core box (half extent - convex radius = 0.45) inflated by the 0.05 convex
				// radius, so the core only covers the cells whose centers are within [-0.45, 0.45] on x
				// and z. The outer ring of cell centers (at +/-0.46875) is offset from the core in x/z, so
				// those contacts are shallower and their axis is diagonal. The deepest contact therefore
				// does not have to come first: it comes from the cells that are directly under the box and
				// its depth is the full 0.05 overlap.
				float max_depth = -FLT_MAX;
				for (const CollideShapeResult &h : collector.mHits)
					max_depth = max(max_depth, h.mPenetrationDepth);
				Trace("box overlap 0.05: %d hits, max depth %.5f", int(collector.mHits.size()), max_depth);
				CHECK_APPROX_EQUAL(max_depth, 0.05f, 0.002f);
			}
		}

		// Touched on the side by a regular sphere
		{
			RefConst<Shape> sphere = new SphereShape(cHalfExtent);
			CollideShapeSettings settings;
			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(cube.mShape, Mat44::sIdentity(), sphere, Mat44::sTranslation(Vec3(1.0f - 0.05f, 0, 0)), settings, collector);
			CHECK(collector.mHits.size() > 0);
			if (!collector.mHits.empty())
			{
				// Touching one face of the cube generates one contact per touched cell. The cells at the border
				// of the face are only touched at a corner, so their contact is shallower and its axis is
				// diagonal. With inscribed sphere cells (sUseSphereContacts) no cell is hit head on (a 16-wide
				// grid has no center line): the deepest contacts are the 4 cells closest to the sphere axis,
				// offset by half a cell in both y and z, so the depth is r1 + r2 - |d| instead of the 0.05 m
				// that box cells (or the convex hull of the whole grid) report. This depth deficit at cell
				// corners/edges is the known cost of the sphere approximation.
				float max_depth = -FLT_MAX;
				Vec3 deepest_axis = Vec3::sZero();
				for (const CollideShapeResult &h : collector.mHits)
					if (h.mPenetrationDepth > max_depth)
					{
						max_depth = h.mPenetrationDepth;
						deepest_axis = h.mPenetrationAxis.Normalized();
					}
				// The nearest cell center along the sphere axis is the center of the outermost face cell,
				// i.e. at (cHalfExtent - half_cell); the sphere is centered at 2 * cHalfExtent - 0.05.
				float half_cell = 0.5f * cVoxelSize;
				float axial = (2.0f * cHalfExtent - 0.05f) - (cHalfExtent - half_cell);
				float deepest = (cHalfExtent + half_cell) - sqrtf(axial * axial + 2.0f * half_cell * half_cell);
				Trace("sphere touch: %d hits, max depth %.5f, expected deepest %.5f", int(collector.mHits.size()), max_depth, deepest);
				CHECK_APPROX_EQUAL(max_depth, deepest, 0.002f);
				CHECK(abs(deepest_axis.GetX()) > 0.98f);

				// No contact may be deeper than the actual overlap of the two shapes: the sphere is 0.05 m
				// inside the cube, so no individual cell may report more than that
				for (const CollideShapeResult &h : collector.mHits)
					CHECK(h.mPenetrationDepth <= 0.05f + 0.002f);
			}
		}
	}

	/// Colliding a voxel grid against any other shape (convex shape, mesh, height field, plane) merges each
	/// run of consecutive solid voxels along X into a single capsule: a capsule of
	/// radius 0.5 cells covers exactly one cell, and a cylinder half
	/// height of (L - 1) / 2 cells makes it cover L of them (the sphere centers land on the cell centers).
	/// One capsule per run means the fixed cost of a shape query (dispatch, mesh tree traversal, result
	/// construction) is paid once per run instead of once per cell -- that is where the win comes from, see
	/// VoxelShapeMeshRunPrimitiveCost for the raw per-primitive costs of the same trade off.
	TEST_CASE("VoxelShapeRunMerge")
	{
		CollideShapeSettings settings;
		settings.mMaxSeparationDistance = 0.0f;

		// A horizontal 20 x 1 x 1 cell bar (2 m long, one cell thick) = one run of 20 cells
		VoxelObject bar(20, 1, 1, cVoxelSize);
		for (int x = 0; x < 20; ++x)
			bar.Cell(x, 0, 0) = 1;
		bar.Update();

		// A box that covers the whole bar, its bottom face 0.5 cells inside the bar.
		//
		// It is deliberately a bit wider than the bar: every convex shape is a core shape (without its
		// convex radius, here 0.05 m for the box, see cDefaultConvexRadius) that is inflated by the radius.
		// With a box half extent of 1.0 the core would span x in [-0.95, 0.95], i.e. the outermost cell
		// centers would land exactly on the core's side faces and the queried distance would be 0 for them
		// (see the non-uniform case below for why that is a problem). Making the box 1.1 wide puts all cell
		// centers strictly inside the core in x and z, so the only way a cell can be outside of it is y.
		RefConst<Shape> box = new BoxShape(Vec3(1.1f, 0.5f, 0.5f));
		Mat44 box_transform = Mat44::sTranslation(Vec3(0, 0.5f, 0));

		{
			// 20 solid cells, but a single run -> a single capsule -> a single contact instead of 20
			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(bar.mShape, Mat44::sIdentity(), box, box_transform, settings, collector);

			Trace("run of 20 cells: %d contacts", int(collector.mHits.size()));
			CHECK(collector.mHits.size() == 1);
			if (collector.mHits.size() == 1)
				CHECK_APPROX_EQUAL(collector.mHits[0].mPenetrationDepth, 0.5f * cVoxelSize, 0.002f);
		}

		{
			// Non-uniform scale: the cells cannot be merged (a capsule only allows a uniform scale), so the
			// query falls back to one inscribed sphere per cell and the contact count goes back up. The
			// scaled grid is 0.5 cells thick in Z, so its spheres have a radius of 0.5 * min(cell size)
			// = 0.25 * cVoxelSize, and the cell centers stay at y = 0.
			//
			// * This also guards the transform convention of the per cell fallback: the scale of the cells is
			//   handed to the narrow phase as the *shape* scale, so the transform of a cell must be a pure
			//   rotation + translation. If the voxel scale ends up in its 3x3 part, CollideConvexVsConvex
			//   inverts it with InversedRotationTranslation (a transpose, correct only for a rotation) and the
			//   query is computed in a wrong space: with a non-uniform cell scale that silently produced 0
			//   contacts for every cell.
			// * The box must not be pushed down until the sphere centers reach its core face (y = 0.05, i.e.
			//   the box at y = 0.45): the sphere core is a single point, so a center that lies exactly on the
			//   core face gives a GJK closest distance of exactly 0, which is the "Indeterminate" case that
			//   hands over to EPA - and EPA needs the origin to be strictly inside the Minkowski difference,
			//   which it is not. That would be "no contact" for all 20 cells again. Keeping the box at y = 0.5
			//   puts the centers half a cell below the core face, a clean 0.25 * cVoxelSize penetration per cell.
			AllHitCollisionCollector<CollideShapeCollector> collector;
			CollisionDispatch::sCollideShapeVsShape(bar.mShape, box, Vec3(1.0f, 1.0f, 0.5f), Vec3::sOne(), Mat44::sIdentity(), box_transform, SubShapeIDCreator(), SubShapeIDCreator(), settings, collector);

			Trace("non-uniform scale: %d contacts", int(collector.mHits.size()));
			CHECK(collector.mHits.size() == 20);
			for (const CollideShapeResult &h : collector.mHits)
				CHECK_APPROX_EQUAL(h.mPenetrationDepth, 0.25f * cVoxelSize, 0.002f);
		}

	// The merge happens on the mesh / height field / convex paths alike. The bar is lifted by 0.3 of a
	// cell so that the row of cell centers sits 0.3 cells above the floor plane and the capsule bottom
	// is 0.2 cells inside of it: a shape whose center lies exactly in the plane of a mesh triangle is a
	// knife edge case (CollideShapeSettings::mBackFaceMode defaults to IgnoreBackFaces, and the sign of
	// the center/plane distance is then decided by rounding) - true for the per cell spheres as well.
	Mat44 bar_lifted = Mat44::sTranslation(Vec3(0, 0.3f * cVoxelSize, 0));

		{
			RefConst<Shape> mesh = sCreateMeshFloor();
			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(bar.mShape, bar_lifted, mesh, Mat44::sIdentity(), settings, collector);

			Trace("run of 20 cells vs mesh: %d contacts", int(collector.mHits.size()));
			CHECK(collector.mHits.size() > 0);
			CHECK(collector.mHits.size() <= 8);					// One capsule against 2 triangles, not 20 cells
			for (const CollideShapeResult &h : collector.mHits)
				CHECK(h.mPenetrationDepth <= 0.2f * cVoxelSize + 0.002f);
		}

		{
			RefConst<Shape> height_field = sCreateHeightFieldFloor();
			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(bar.mShape, bar_lifted, height_field, Mat44::sIdentity(), settings, collector);

			Trace("run of 20 cells vs height field: %d contacts", int(collector.mHits.size()));
			CHECK(collector.mHits.size() > 0);
			CHECK(collector.mHits.size() <= 16);
			for (const CollideShapeResult &h : collector.mHits)
				CHECK(h.mPenetrationDepth <= 0.2f * cVoxelSize + 0.002f);
		}

		{
			RefConst<Shape> plane = new PlaneShape(Plane(Vec3::sAxisY(), 0.0f));
			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(bar.mShape, bar_lifted, plane, Mat44::sIdentity(), settings, collector);

			Trace("run of 20 cells vs plane: %d contacts", int(collector.mHits.size()));
			CHECK(collector.mHits.size() > 0);
			CHECK(collector.mHits.size() <= 4);
			for (const CollideShapeResult &h : collector.mHits)
				CHECK(h.mPenetrationDepth <= 0.2f * cVoxelSize + 0.002f);
		}

		{
			// An empty cell splits the run: a run of 10 cells (x = 0..9) plus a run of 9 (x = 11..19)
			// -> two capsules. (A run is now simply the longest stretch of consecutive solid cells -
			// the voxel no longer carries a material, so there is no per voxel value left to break
			// a run on.)
			bar.Cell(10, 0, 0) = 0;
			bar.Update();

			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(bar.mShape, Mat44::sIdentity(), box, box_transform, settings, collector);

			Trace("two runs (10 + 9 cells) of the bar: %d contacts", int(collector.mHits.size()));
			CHECK(collector.mHits.size() == 2);
		}

		{
			// Runs longer than the capsule pool (64 cells) are split: 130 cells -> 64 + 64 + 2 -> 3 capsules
			constexpr int cLong = 130;
			VoxelObject long_bar(cLong, 1, 1, cVoxelSize);
			for (int x = 0; x < cLong; ++x)
				long_bar.Cell(x, 0, 0) = 1;
			long_bar.Update();

			RefConst<Shape> long_box = new BoxShape(Vec3(0.5f * cLong * cVoxelSize, 0.5f, 0.5f));
			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(long_bar.mShape, Mat44::sIdentity(), long_box, box_transform, settings, collector);

			Trace("run of %d cells: %d contacts", cLong, int(collector.mHits.size()));
			CHECK(collector.mHits.size() == 3);
		}

	}

	TEST_CASE("VoxelShapeDropOnFloor")
	{
		// Cube, sphere and cylinder are all dropped on a box floor, a mesh floor and a height field floor.
		// The lowest voxel of all three shapes is at -0.5 so the body comes to rest with its center at y = 0.5
		for (int floor_type = 0; floor_type < 3; ++floor_type)
			for (int shape_type = 0; shape_type < 3; ++shape_type)
			{
				PhysicsTestContext c;
				switch (floor_type)
				{
				case 0:
					c.CreateFloor();
					break;

				case 1:
					c.CreateBody(BodyCreationSettings(sCreateMeshFloor(), RVec3::sZero(), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);
					break;

				case 2:
					c.CreateBody(BodyCreationSettings(sCreateHeightFieldFloor(), RVec3::sZero(), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);
					break;
				}

				VoxelObject object(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
				sMakeShape(object, shape_type);

				BodyID id = c.GetBodyInterface().CreateAndAddBody(BodyCreationSettings(object.mShape, RVec3(0, 2.0, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

				c.Simulate(3.0f);

				CAPTURE(floor_type);
				CAPTURE(shape_type);
				sCheckApproxEqual(c.GetBodyInterface().GetPosition(id).GetY(), cHalfExtent, 0.05);

				// It should be resting level, not lying on its side
				Quat rotation = c.GetBodyInterface().GetRotation(id);
				CAPTURE(rotation.GetX());
				CAPTURE(rotation.GetZ());
				CHECK(abs(rotation.GetX()) < 0.05f);
				CHECK(abs(rotation.GetZ()) < 0.05f);
			}
	}

	TEST_CASE("VoxelShapeDestruction")
	{
		PhysicsTestContext c;
		c.CreateFloor();

		VoxelObject cube(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCube(cube);

		RefConst<Shape> shape = cube.mShape;
		const VoxelShape *voxel_shape = static_cast<const VoxelShape *>(shape.GetPtr());
		float original_mass = shape->GetMassProperties().mMass;

		BodyInterface &bi = c.GetBodyInterface();
		BodyID id = bi.CreateAndAddBody(BodyCreationSettings(shape, RVec3(0, cHalfExtent, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

		// Let it settle on the floor
		c.Simulate(1.0f);
		sCheckApproxEqual(bi.GetPosition(id).GetY(), cHalfExtent, 0.02);

		// Remove the bottom layer of voxels. The shape reads the voxel data directly so calling Recalculate is
		// enough to update the collision geometry, no shape and no body rebuild is needed.
		for (int z = 0; z < cNumVoxels; ++z)
			for (int x = 0; x < cNumVoxels; ++x)
				cube.Cell(x, 0, z) = 0;

		bi.ActivateBody(id);

		// The mass properties are cached, they only change when we ask for them to be recalculated. Note that
		// this is purely a cache: the collision queries read the voxel array directly, so the contacts were
		// already correct before this call.
		CHECK_APPROX_EQUAL(shape->GetMassProperties().mMass, original_mass, 1.0e-3f);
		voxel_shape->Recalculate();
		CHECK_APPROX_EQUAL(shape->GetMassProperties().mMass, original_mass * (cNumVoxels - 1.0f) / cNumVoxels, 1.0e-2f);
		CHECK(voxel_shape->GetNumSolidVoxels() == cNumVoxels * cNumVoxels * (cNumVoxels - 1));

		// The bottom of the shape is now one voxel higher so the body falls by one voxel size
		c.Simulate(1.0f);
		sCheckApproxEqual(bi.GetPosition(id).GetY(), cHalfExtent - cVoxelSize, 0.02);
	}

	/// Reports the cost of the narrow phase for voxel shapes and compares it against the equivalent box shapes.
	/// This is not a correctness test (it has no assertions), it exists to be able to judge the cost of the
	/// collision queries.
	///
	/// A voxel shape has no derived geometry (no hull, no mesh): the narrow phase walks the solid voxels in the
	/// overlap region. So the cost of a voxel vs voxel query is (number of solid voxels in
	/// the overlap) x (cost of one filtered cell pair), which is exactly what this table shows: compare a 10^3
	/// voxel cube against a 20^3 and a 32^3 one and the per query cost should grow with the number of voxels on
	/// the touching faces, while the box vs box line is the floor that a single cell can get to.
	///
	/// Note that this measures the collision query itself, not a simulation step: Jolt only re-runs the narrow
	/// phase for bodies that are awake (and it stops calling it once a stack has settled and went to sleep),
	/// so timing simulation steps would not say anything about the cost of the individual query.
	TEST_CASE("VoxelShapePerformance")
	{
		constexpr int cNumQueries = 1000;

		// Query a shape that is penetrating a static body with the same shape, see the note above
		auto measure = [&](const Shape *inShape, const char *inName, float inExtent, float inFraction, ECollectFacesMode inFacesMode) {
			PhysicsTestContext c;

			// Static body that we are going to collide with. The query shape is the same shape, so for the
			// voxel case this measures voxel vs voxel.
			c.CreateBody(BodyCreationSettings(inShape, RVec3::sZero(), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

			// Settings used when generating contacts (this is what PhysicsSystem does too). Collecting the faces
			// makes the narrow phase call Shape::GetSupportingFace, not collecting them isolates the cost of
			// GJK/EPA from the cost of building the contact manifold.
			CollideShapeSettings settings;
			settings.mMaxSeparationDistance = 0.0f;
			settings.mCollectFacesMode = inFacesMode;

			// Query shape, penetrating the target by a fraction of its size
			RMat44 query_transform = RMat44::sRotationTranslation(Quat::sRotation(Vec3(1, 1, 0.5f).Normalized(), DegreesToRadians(10.0f)), RVec3(0, inFraction * inExtent, 0));

			// Warm up
			{
				AllHitCollisionCollector<CollideShapeCollector> collector;
				c.GetSystem()->GetNarrowPhaseQuery().CollideShape(inShape, Vec3::sOne(), query_transform, settings, RVec3::sZero(), collector);
			}

			chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
			for (int i = 0; i < cNumQueries; ++i)
			{
				AllHitCollisionCollector<CollideShapeCollector> collector;
				c.GetSystem()->GetNarrowPhaseQuery().CollideShape(inShape, Vec3::sOne(), query_transform, settings, RVec3::sZero(), collector);
			}
			chrono::microseconds duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);

			Trace("%s: %.3f us / query", inName, double(duration.count()) / cNumQueries);
		};

		RefConst<Shape> box = new BoxShape(Vec3::sReplicate(cHalfExtent));

		// A planar contact (tilted by 10 degrees) and a deep penetration
		measure(box, "BoxShape, resting contact      ", cNumVoxels * cVoxelSize, 0.98f, ECollectFacesMode::CollectFaces);
		measure(box, "BoxShape, deep penetration     ", cNumVoxels * cVoxelSize, 0.70f, ECollectFacesMode::CollectFaces);

		// The same voxel cube at 3 grid resolutions. The narrow phase is per voxel, so the cost scales with the
		// number of voxels that overlap the other shape, i.e. with the area of the touching faces.
		// "no faces" skips the contact manifold generation and shows how much of the time is the per cell
		// convex vs convex query itself (see the "CollectFaces" note above).
		{
			VoxelObject object(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
			sMakeCube(object);
			measure(object.mShape, "VoxelShape  10^3, resting      ", cNumVoxels * cVoxelSize, 0.98f, ECollectFacesMode::CollectFaces);
			measure(object.mShape, "VoxelShape  10^3, deep         ", cNumVoxels * cVoxelSize, 0.70f, ECollectFacesMode::CollectFaces);
			measure(object.mShape, "VoxelShape  10^3, no faces     ", cNumVoxels * cVoxelSize, 0.98f, ECollectFacesMode::NoFaces);
		}
		{
			constexpr int cSize = 20;
			VoxelObject object(cSize, cSize, cSize, cVoxelSize);
			sMakeCube(object);
			measure(object.mShape, "VoxelShape  20^3, resting      ", cSize * cVoxelSize, 0.98f, ECollectFacesMode::CollectFaces);
			measure(object.mShape, "VoxelShape  20^3, no faces     ", cSize * cVoxelSize, 0.98f, ECollectFacesMode::NoFaces);
		}
		{
			constexpr int cSize = 32;
			VoxelObject object(cSize, cSize, cSize, cVoxelSize);
			sMakeCube(object);
			measure(object.mShape, "VoxelShape  32^3, resting      ", cSize * cVoxelSize, 0.98f, ECollectFacesMode::CollectFaces);
			measure(object.mShape, "VoxelShape  32^3, no faces     ", cSize * cVoxelSize, 0.98f, ECollectFacesMode::NoFaces);
		}
	}

	/// Measures how the cost of a collision query scales with the size of the grid. The narrow phase only
	/// visits the solid voxels in the overlap of the grid with the other shape (see sVisitSolidVoxelsInBox),
	/// so a query against a small shape must cost the same no matter how big the grid is. That is what makes a
	/// large destructible world possible: the cost of colliding with a wall does not depend on the size of the
	/// map, only on the amount of overlap.
	TEST_CASE("VoxelShapeQueryScaling")
	{
		constexpr int cNumQueries = 2000;

		// The query shape is a small box that penetrates the surface of the grid by 5 cm
		RefConst<Shape> probe = new BoxShape(Vec3::sReplicate(0.1f));

		CollideShapeSettings settings;
		settings.mMaxSeparationDistance = 0.0f;

		for (int size : { 10, 20, 32, 64 })
		{
			VoxelObject object(size, size, size, cVoxelSize);
			sMakeCube(object);

			float half_extent = 0.5f * size * cVoxelSize;
			Mat44 probe_transform = Mat44::sTranslation(Vec3(0.0f, half_extent - 0.05f, 0.0f));

			auto query = [&]() {
				AllHitCollisionCollector<CollideShapeCollector> collector;
				CollisionDispatch::sCollideShapeVsShape(
					probe, object.mShape,
					Vec3::sOne(), Vec3::sOne(),
					probe_transform, Mat44::sIdentity(),
					SubShapeIDCreator(), SubShapeIDCreator(),
					settings, collector);
				return (int)collector.mHits.size();
			};

			// Warm up and check that the query actually finds something
			int hits = query();
			CHECK(hits > 0);

			chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
			int checksum = 0;
			for (int i = 0; i < cNumQueries; ++i)
				checksum += query();
			chrono::microseconds duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - start);

			CHECK(checksum == hits * cNumQueries);
			Trace("Voxel %3d^3 (%7d solid voxels): %6.2f us / query (%d hits)", size,
				static_cast<const VoxelShape *>(object.mShape.GetPtr())->GetNumSolidVoxels(),
				double(duration.count()) / cNumQueries, hits);
		}
	}

	/// Cast a convex shape against a voxel shape. This is the path that the character controller and the CCD
	/// use and it is the one path that a collide query cannot express. It has to walk the cells in the swept
	/// bounding box of the cast and cast the shape against each of them, so the cast has to be moved to the
	/// center of mass of every cell (a translation in the frame of the voxel shape, not in the frame of the
	/// cast, which is rotated with the shape that is being cast).
	TEST_CASE("VoxelShapeCast")
	{
		constexpr int cSize = 20;
		constexpr int cWallMin = 8;
		constexpr int cWallMax = 12;
		constexpr int cHoleMin = 4;
		constexpr int cHoleMax = 16;

		// The same wall with a hole as in VoxelShapeConcave: 1.25 x 0.25 x 1.25 m, centered on the origin,
		// with a 0.75 x 0.25 x 0.75 m hole through it
		VoxelObject wall(cSize, cSize, cSize, cVoxelSize);
		for (int z = 0; z < cSize; ++z)
			for (int x = 0; x < cSize; ++x)
				for (int y = cWallMin; y < cWallMax; ++y)
				{
					bool in_hole = x >= cHoleMin && x < cHoleMax && z >= cHoleMin && z < cHoleMax;
					if (!in_hole)
						wall.Cell(x, y, z) = 1;
				}
		wall.Update();

		// A 0.3 m box, it fits through the hole (the hole is 0.75 m wide and 0.25 m tall)
		RefConst<Shape> probe = new BoxShape(Vec3::sReplicate(0.15f));
		ShapeCastSettings settings;

		// The cast is expressed relative to the center of mass of the voxel shape (that is what
		// sCastShapeVsShapeLocalSpace expects) and the voxel shape is at the origin in this test
		auto cast = [&](Vec3Arg inStart, Vec3Arg inDirection) {
			ClosestHitCollisionCollector<CastShapeCollector> collector;
			ShapeCast shape_cast(probe, Vec3::sOne(), Mat44::sTranslation(inStart), inDirection);
			CollisionDispatch::sCastShapeVsShapeLocalSpace(shape_cast, settings, wall.mShape, Vec3::sOne(), ShapeFilter(), Mat44::sIdentity(), SubShapeIDCreator(), SubShapeIDCreator(), collector);
			return collector;
		};

		// Through the hole: the hole is a shaft through the wall in the y direction (the wall is a slab of
		// 0.25 m in y), so the cast has to run along y to pass through it. The box starts 1 m above the wall
		// and travels 2 m, ending 1 m below it.
		{
			ClosestHitCollisionCollector<CastShapeCollector> collector = cast(Vec3(0, 1.0f, 0), Vec3(0, -2.0f, 0));
			CHECK(!collector.HadHit());
		}

		// Next to the hole (above the solid part of the wall) the box hits the top face of the wall. The wall's
		// top face is at y = cWallMax * cVoxelSize - 0.5 * cSize * cVoxelSize and the box is 0.15 m, so the box
		// travels 1.0 - 0.15 - wall_top out of the 2 m.
		{
			ClosestHitCollisionCollector<CastShapeCollector> collector = cast(Vec3(-0.5f, 1.0f, 0), Vec3(0, -2.0f, 0));
			CHECK(collector.HadHit());
			if (collector.HadHit())
			{
				float wall_top = cWallMax * cVoxelSize - 0.5f * cSize * cVoxelSize;
				CHECK_APPROX_EQUAL(collector.mHit.mFraction, (1.0f - 0.15f - wall_top) / 2.0f, 0.005f);
				CHECK(abs(collector.mHit.mPenetrationAxis.Normalized().GetY()) > 0.99f);
			}
		}

		// Next to the hole in the other direction too: the box hits the front face of the wall. The face is at
		// z = -0.5 * cSize * cVoxelSize and the box is 0.15 m, so the box travels 1.5 - 0.15 - |z_face| out of
		// the 2 m.
		{
			ClosestHitCollisionCollector<CastShapeCollector> collector = cast(Vec3(-0.5f, 0, -1.5f), Vec3(0, 0, 2.0f));
			CHECK(collector.HadHit());
			if (collector.HadHit())
			{
				float wall_front = -0.5f * cSize * cVoxelSize;
				CHECK_APPROX_EQUAL(collector.mHit.mFraction, (1.5f - 0.15f + wall_front) / 2.0f, 0.005f);
				CHECK(abs(collector.mHit.mPenetrationAxis.Normalized().GetZ()) > 0.99f);
			}
		}

		// A rotated cast hits the wall too (this is the case that breaks when the cell offset is applied in the
		// frame of the cast instead of the frame of the voxel shape)
		{
			ClosestHitCollisionCollector<CastShapeCollector> collector;
			ShapeCast shape_cast(probe, Vec3::sOne(), Mat44::sRotationTranslation(Quat::sRotation(Vec3::sAxisY(), DegreesToRadians(45.0f)), Vec3(-1.5f, 0, -1.5f)), Vec3(2.0f, 0, 2.0f));
			CollisionDispatch::sCastShapeVsShapeLocalSpace(shape_cast, settings, wall.mShape, Vec3::sOne(), ShapeFilter(), Mat44::sIdentity(), SubShapeIDCreator(), SubShapeIDCreator(), collector);
			CHECK(collector.HadHit());
			if (collector.HadHit())
				CHECK(collector.mHit.mFraction < 1.0f);
		}

		// The wall is only 0.25 m tall, a box that is cast above it does not hit anything
		{
			ClosestHitCollisionCollector<CastShapeCollector> collector = cast(Vec3(-0.5f, 0.5f, -1.5f), Vec3(0, 0, 2.0f));
			CHECK(!collector.HadHit());
		}
	}

	/// Regression test: the DDA used to report hit fractions relative to the slab-clipped start
	/// point instead of the ray origin, so any ray that entered the grid from outside and flew
	/// through empty cells before the hit reported a fraction that was too small by t_enter.
	/// (A ray that hits the very first cell it enters was unaffected, which is why the other
	/// ray tests did not catch this.)
	TEST_CASE("VoxelShapeRayFraction")
	{
		constexpr int cSize = 10;

		// A wall with a hole through it: 2 cells thick in z, hole at x, y in [2, 6). The wall layers are
		// symmetric about the grid center, so their faces sit exactly one cell from the center.
		VoxelObject wall(cSize, cSize, cSize, cVoxelSize);
		for (int z = 4; z <= 5; ++z)
			for (int y = 0; y < cSize; ++y)
				for (int x = 0; x < cSize; ++x)
				{
					bool in_hole = x >= 2 && x < 6 && y >= 2 && y < 6;
					if (!in_hole)
						wall.Cell(x, y, z) = 1;
				}
		wall.Update();

		// Through the hole: the ray starts outside the grid and never leaves the hole, no hit
		{
			RayCastResult hit;
			CHECK(!wall.mShape->CastRay(RayCast(Vec3(-0.1f, -0.1f, -1.0f), Vec3(0, 0, 2.0f)), SubShapeIDCreator(), hit));
		}

		// Solid part: the front face of the wall is at z = -cVoxelSize, the ray starts at z = -1 m
		// with a 2 m direction, so the fraction is (1.0 - cVoxelSize) / 2.0
		// (before the fix this was missing t_enter, so it came out too small)
		{
			RayCastResult hit;
			CHECK(wall.mShape->CastRay(RayCast(Vec3(0.3f, 0.3f, -1.0f), Vec3(0, 0, 2.0f)), SubShapeIDCreator(), hit));
			CHECK_APPROX_EQUAL(hit.mFraction, (1.0f - cVoxelSize) / 2.0f, 0.005f);
		}

		// The same ray in the opposite direction: the back face of the wall is at z = +cVoxelSize
		// and the ray starts at z = 1 m, so the fraction is (1.0 - cVoxelSize) / 2.0 as well
		{
			RayCastResult hit;
			CHECK(wall.mShape->CastRay(RayCast(Vec3(0.3f, 0.3f, 1.0f), Vec3(0, 0, -2.0f)), SubShapeIDCreator(), hit));
			CHECK_APPROX_EQUAL(hit.mFraction, (1.0f - cVoxelSize) / 2.0f, 0.005f);
		}

		// Unit-length direction: start at z = -1 m, front face at z = -cVoxelSize => fraction 1 - cVoxelSize
		{
			RayCastResult hit;
			CHECK(wall.mShape->CastRay(RayCast(Vec3(0.3f, 0.3f, -1.0f), Vec3(0, 0, 1.0f)), SubShapeIDCreator(), hit));
			CHECK_APPROX_EQUAL(hit.mFraction, 1.0f - cVoxelSize, 0.005f);
		}
	}

	/// A voxel shape collides as the voxels are drawn, not as the convex hull of the voxels. A wall with a hole
	/// in it must let a box travel through the hole: this is the property that makes a voxel shape usable for a
	/// destructive environment (a window in a wall, a tunnel dug through the terrain, a building that breaks
	/// apart). With the convex hull of the voxels the hole is filled in and the box can never pass, which is
	/// why the shape is no longer registered as a convex shape.
	TEST_CASE("VoxelShapeConcave")
	{
		constexpr int cSize = 20;
		constexpr int cWallMin = 8;
		constexpr int cWallMax = 12;
		constexpr int cHoleMin = 4;
		constexpr int cHoleMax = 16;

		VoxelObject wall(cSize, cSize, cSize, cVoxelSize);

		// A wall (4 voxels thick) with a square hole (12 x 4 x 12 voxels, so 0.75 x 0.25 x 0.75 m) through it
		for (int z = 0; z < cSize; ++z)
			for (int x = 0; x < cSize; ++x)
				for (int y = cWallMin; y < cWallMax; ++y)
				{
					bool in_hole = x >= cHoleMin && x < cHoleMax && z >= cHoleMin && z < cHoleMax;
					if (!in_hole)
						wall.Cell(x, y, z) = 1;
				}
		wall.Update();

		// The hole is empty, everything else of the wall slab is solid
		int expected_solid = (cWallMax - cWallMin) * (cSize * cSize - (cHoleMax - cHoleMin) * (cHoleMax - cHoleMin));
		CHECK(wall.CountSolid() == expected_solid);
		CHECK(static_cast<const VoxelShape *>(wall.mShape.GetPtr())->GetNumSolidVoxels() == expected_solid);

		// A box (0.3 m) that fits through the hole
		RefConst<Shape> probe = new BoxShape(Vec3::sReplicate(0.15f));

		CollideShapeSettings settings;
		settings.mMaxSeparationDistance = 0.0f;

		auto num_hits = [&](Vec3Arg inPosition) {
			AllHitCollisionCollector<CollideShapeCollector> collector;
			CollisionDispatch::sCollideShapeVsShape(
				probe, wall.mShape,
				Vec3::sOne(), Vec3::sOne(),
				Mat44::sTranslation(inPosition), Mat44::sIdentity(),
				SubShapeIDCreator(), SubShapeIDCreator(),
				settings, collector);
			return (int)collector.mHits.size();
		};

		// The center of the hole is at the center of the grid
		CHECK(num_hits(Vec3::sZero()) == 0);

		// The same box next to the hole, inside the solid slab of the wall, must collide
		CHECK(num_hits(Vec3(-0.5f, 0.0f, 0.0f)) > 0);

		// The hole goes through the whole wall, so a 0.3 m box travels through it in the other direction as
		// well. The hole is 0.75 m wide, the box is 0.3 m, so the box can move 0.2 m to either side without
		// touching the wall. This is what the convex hull could never do: the hull of this wall fills the hole
		// completely, so with a convex shape the box would already collide at z = 0.
		for (float z = -0.2f; z <= 0.2f; z += 0.05f)
		{
			CAPTURE(z);
			CHECK(num_hits(Vec3(0.0f, 0.0f, z)) == 0);
		}

		// The same path through the solid part of the wall (next to the hole) does hit it, so the hole is a
		// real hole and not "this slice of the wall is empty"
		for (float z = -0.2f; z <= 0.2f; z += 0.05f)
		{
			CAPTURE(z);
			CHECK(num_hits(Vec3(-0.5f, 0.0f, z)) > 0);
		}

		// A box that straddles the edge of the hole collides with the solid part of the wall next to it
		CHECK(num_hits(Vec3(-0.35f, 0.0f, 0.0f)) > 0);
		CHECK(num_hits(Vec3(0.0f, 0.0f, -0.35f)) > 0);

		// And outside of the wall slab (which is only 1.25 x 1.25 x 0.25 m in x, z, y) there is nothing to hit
		CHECK(num_hits(Vec3(0.0f, 0.0f, 1.3f)) == 0);
		CHECK(num_hits(Vec3(-0.5f, 0.0f, 1.3f)) == 0);
		CHECK(num_hits(Vec3(-0.5f, 0.5f, 0.0f)) == 0);
	}


#ifdef JPH_DEBUG_RENDERER
	/// Debug renderer that collects the triangles that a shape emits. It allows verifying the generated debug
	/// geometry (which is normally uploaded to the GPU) without a graphics device.
	class CollectingDebugRenderer final : public DebugRendererSimple
	{
	public:
		/// A triangle in world space
		struct Tri
		{
			Vec3			mV0, mV1, mV2;
		};

		virtual void		DrawLine([[maybe_unused]] RVec3Arg inFrom, [[maybe_unused]] RVec3Arg inTo, [[maybe_unused]] ColorArg inColor) override { }
		virtual void		DrawText3D([[maybe_unused]] RVec3Arg inPosition, [[maybe_unused]] const string_view &inString, [[maybe_unused]] ColorArg inColor, [[maybe_unused]] float inHeight) override { }
		virtual void		DrawTriangle(RVec3Arg inV1, RVec3Arg inV2, RVec3Arg inV3, [[maybe_unused]] ColorArg inColor, [[maybe_unused]] DebugRenderer::ECastShadow inCastShadow) override
		{
			mTriangles.push_back({ Vec3(inV1), Vec3(inV2), Vec3(inV3) });
		}

		Array<Tri>			mTriangles;
	};

	/// Verify the debug geometry that VoxelShape::Draw builds: only the exposed faces of the voxel grid are drawn
	/// (an interior face can never be seen), the winding is counter clockwise when looked at from the outside
	/// (otherwise the face would be culled by the renderer) and the geometry stays inside the bounds of the shape.
	TEST_CASE("VoxelShapeDebugGeometry")
	{
		// A solid cube has 6 sides of 10 x 10 exposed faces
		{
			VoxelObject cube(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
			sMakeCube(cube);

			CollectingDebugRenderer renderer;
			cube.mShape->Draw(&renderer, RMat44::sIdentity(), Vec3::sOne(), Color::sWhite, false, false);

			// 6 sides * 10 * 10 quads * 2 triangles
			CHECK(renderer.mTriangles.size() == size_t(6 * cNumVoxels * cNumVoxels * 2));
		}

		// All shapes: check the winding and the bounds
		for (int shape_type = 0; shape_type < 3; ++shape_type)
		{
			VoxelObject object(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
			sMakeShape(object, shape_type);

			CollectingDebugRenderer renderer;
			object.mShape->Draw(&renderer, RMat44::sIdentity(), Vec3::sOne(), Color::sWhite, false, false);

			CHECK(!renderer.mTriangles.empty());

			// All 3 test shapes are convex and centered on the origin of the shape, so the winding of a triangle is
			// correct if the normal that follows from it points away from the center
			int wrong_winding = 0, degenerate = 0, out_of_bounds = 0;
			for (const CollectingDebugRenderer::Tri &t : renderer.mTriangles)
			{
				Vec3 normal = (t.mV1 - t.mV0).Cross(t.mV2 - t.mV0);
				if (normal.LengthSq() < 1.0e-12f)
					++degenerate;
				else if (normal.Dot((t.mV0 + t.mV1 + t.mV2) / 3.0f) <= 0.0f)
					++wrong_winding;

				for (const Vec3 &v : { t.mV0, t.mV1, t.mV2 })
					if (abs(v.GetX()) > cHalfExtent + 1.0e-5f || abs(v.GetY()) > cHalfExtent + 1.0e-5f || abs(v.GetZ()) > cHalfExtent + 1.0e-5f)
						++out_of_bounds;
			}
			CHECK(wrong_winding == 0);
			CHECK(degenerate == 0);
			CHECK(out_of_bounds == 0);

			Trace("debug geometry of shape %d: %u triangles", shape_type, (uint)renderer.mTriangles.size());
		}
	}
#endif // JPH_DEBUG_RENDERER

	/// Incremental voxel edits (UpdateSolid / UpdateSolids) must leave the shape in exactly the same state as
	/// editing the data and calling Recalculate() afterwards: same cached counts, same mass properties and -
	/// the part that really matters - bit identical collision results. The collision comparison is what proves
	/// the two things the incremental path maintains by hand: the occupancy bit itself (one O(1) flip in the
	/// bitmap that the traversal reads directly, no derived index involved) and the conservative solid
	/// bounding box (it only grows while editing, so it may be larger than the exact one until the next
	/// Recalculate()).
	TEST_CASE("VoxelShapeIncrementalUpdateEquivalence")
	{
		// A 13^3 grid crosses the 8x8x8 chunk and 4x4x4 sub chunk boundaries along every axis
		constexpr int cSize = 13;

		VoxelObject incremental(cSize, cSize, cSize, cVoxelSize);	// edited through UpdateSolid / UpdateSolids
		VoxelObject reference(cSize, cSize, cSize, cVoxelSize);		// edited through Cell() + Recalculate()
		sMakeCube(incremental);
		sMakeCube(reference);

		const VoxelShape *incremental_shape = static_cast<const VoxelShape *>(incremental.mShape.GetPtr());
		const VoxelShape *reference_shape = static_cast<const VoxelShape *>(reference.mShape.GetPtr());

		// Deterministic LCG so a failure is reproducible
		uint32 state = 0x12345678u;
		auto next = [&state]()
		{
			state = state * 1664525u + 1013904223u;
			return state;
		};

		CollideShapeSettings settings;
		settings.mMaxSeparationDistance = 0.02f;
		settings.mCollectFacesMode = ECollectFacesMode::NoFaces;
		Mat44 transform = Mat44::sRotationTranslation(Quat::sRotation(Vec3(1, 1, 0.5f).Normalized(), DegreesToRadians(10.0f)), Vec3(0, 0.98f * cSize * cVoxelSize, 0));

		auto collide = [&](const VoxelShape *inShape, AllHitCollisionCollector<CollideShapeCollector> &outCollector)
		{
			sCollide(inShape, transform, inShape, Mat44::sIdentity(), settings, outCollector);
		};

		// The cached counters / mass properties must match the reference, whatever the geometry looks like
		auto check_stats_same = [&]()
		{
			CHECK(incremental_shape->GetNumSolidVoxels() == reference_shape->GetNumSolidVoxels());
			CHECK(incremental.CountSolid() == reference.CountSolid());

			const MassProperties &ma = incremental_shape->GetMassProperties();
			const MassProperties &mb = reference_shape->GetMassProperties();
			CHECK(abs(ma.mMass - mb.mMass) <= 1.0e-3f);
			for (int i = 0; i < 3; ++i)
				CHECK(abs(ma.mInertia(i, i) - mb.mInertia(i, i)) <= 1.0e-3f * max(1.0f, abs(mb.mInertia(i, i))));
		};

		// ... and the collision results must be identical contact for contact
		auto check_same_as_reference = [&](const char *inStage)
		{
			check_stats_same();

			AllHitCollisionCollector<CollideShapeCollector> hits_incremental, hits_reference;
			collide(incremental_shape, hits_incremental);
			collide(reference_shape, hits_reference);
			CHECK(sSameHits(hits_incremental, hits_reference));
			CHECK(hits_reference.mHits.size() > 0);

			Trace("incremental equivalence (%s): %u contacts, %u solid",
				inStage, (uint)hits_reference.mHits.size(), incremental_shape->GetNumSolidVoxels());
		};

		// 300 random edits through a single batched UpdateSolids() call, then 300 more one UpdateSolid() at a
		// time - both must behave exactly like editing the bits and recalculating once
		Array<VoxelShape::SolidEdit> batch;
		for (int i = 0; i < 300; ++i)
		{
			VoxelShape::SolidEdit edit;
			edit.mX = int(next() % cSize);
			edit.mY = int(next() % cSize);
			edit.mZ = int(next() % cSize);
			edit.mSolid = (next() & 1) != 0? 1 : 0;
			batch.push_back(edit);
		}
		for (const VoxelShape::SolidEdit &edit : batch)
			reference.Cell(edit.mX, edit.mY, edit.mZ) = edit.mSolid;

		incremental_shape->UpdateSolids(batch.data(), (int)batch.size());
		reference.Update();
		check_same_as_reference("batched");

		for (int i = 0; i < 300; ++i)
		{
			int x = int(next() % cSize), y = int(next() % cSize), z = int(next() % cSize);
			bool value = (next() & 1) != 0;
			incremental_shape->UpdateSolid(x, y, z, value);
			reference.Cell(x, y, z) = value;
		}
		reference.Update();
		check_same_as_reference("one at a time");

		// Carve a tunnel straight through the grid without ever calling Recalculate(): a cell that became empty
		// must stop being visited immediately (the traversal reads the bitmap itself, so there is nothing to
		// refresh) and it exercises the "bounding box only grows while editing" compromise from another direction
		for (int y = 0; y < cSize; ++y)
		{
			incremental_shape->UpdateSolid(cSize / 2, y, cSize / 2, false);
			reference.Cell(cSize / 2, y, cSize / 2) = 0;
		}
		reference.Update();
		check_same_as_reference("carved tunnel");

		// Delete everything: the incremental path must land on the same empty state as Recalculate()
		for (int z = 0; z < cSize; ++z)
			for (int y = 0; y < cSize; ++y)
				for (int x = 0; x < cSize; ++x)
				{
					incremental_shape->UpdateSolid(x, y, z, false);
					reference.Cell(x, y, z) = 0;
				}
		reference.Update();
		check_stats_same();
		CHECK(incremental_shape->GetNumSolidVoxels() == 0);
		{
			AllHitCollisionCollector<CollideShapeCollector> hits;
			collide(incremental_shape, hits);
			CHECK(hits.mHits.empty());
		}

		// ... and put a single cell back: the shape must come back to life (1 cell is too small to collide
		// with its own rotated copy, so only the counters are compared here)
		incremental_shape->UpdateSolid(cSize / 2, cSize / 2, cSize / 2, true);
		reference.Cell(cSize / 2, cSize / 2, cSize / 2) = 1;
		reference.Update();
		check_stats_same();
		CHECK(incremental_shape->GetNumSolidVoxels() == 1);

		// Finally, a full Recalculate() on the incrementally edited shape must give the exact same answer too -
		// that pins down the "bounding box is conservative until the next Recalculate()" part (the collisions
		// above do not depend on how tight the bounding box is)
		incremental_shape->Recalculate();
		check_stats_same();
	}

	/// How much cheaper is an incremental edit than a full Recalculate()? Both variants write exactly the same
	/// voxels (the same edit list), so the difference is purely the cache maintenance: SetSolid() + Recalculate()
	/// rescans the whole grid, UpdateSolids() touches only the edited cells and their neighbours. The shapes are
	/// rebuilt outside of the timed section so that both variants start from the same state.
	TEST_CASE("VoxelShapeIncrementalUpdateCost")
	{
		enum { cRounds = 9, cNumEdits = 256 };

		auto best = [](const double *inValues)
		{
			double m = inValues[0];
			for (int i = 1; i < cRounds; ++i)
				if (inValues[i] < m)
					m = inValues[i];
			return m;
		};

		for (int size : { 16, 32, 64 })
		{
			// One fixed edit list, shared by both variants
			Array<VoxelShape::SolidEdit> batch;
			uint32 state = 0xabcdef01u + uint32(size);
			auto next = [&state]()
			{
				state = state * 1664525u + 1013904223u;
				return state;
			};
			for (int i = 0; i < cNumEdits; ++i)
			{
				VoxelShape::SolidEdit edit;
				edit.mX = int(next() % size);
				edit.mY = int(next() % size);
				edit.mZ = int(next() % size);
				edit.mSolid = (next() & 1) != 0;
				batch.push_back(edit);
			}

			auto measure_incremental = [&]() -> double
			{
				VoxelObject object(size, size, size, cVoxelSize);
				sMakeCube(object);
				const VoxelShape *shape = static_cast<const VoxelShape *>(object.mShape.GetPtr());
				// The derived statistics (solid count / bounds / mass properties) are lazy, so the first
				// edit would pay for the whole grid. Warm them up before the timer: this benchmark is
				// about the incremental edit cost, not about the one-off full pass.
				shape->GetMassProperties();
				chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();

				shape->UpdateSolids(batch.data(), (int)batch.size());
				chrono::nanoseconds d = chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - start);
				return double(d.count()) * 1.0e-3;			// ns -> us
			};

			auto measure_recalculate = [&]() -> double
			{
				VoxelObject object(size, size, size, cVoxelSize);
				sMakeCube(object);
				const VoxelShape *shape = static_cast<const VoxelShape *>(object.mShape.GetPtr());

				chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
				for (const VoxelShape::SolidEdit &edit : batch)
					shape->SetSolid(edit.mX, edit.mY, edit.mZ, edit.mSolid);
				shape->Recalculate();
				shape->GetMassProperties();					// lazy derived statistics: a rebuild nobody asks them from is not comparable
				chrono::nanoseconds d = chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - start);
				return double(d.count()) * 1.0e-3;			// ns -> us
			};

			double t_incremental[cRounds], t_recalculate[cRounds];
			for (int round = 0; round < cRounds; ++round)
				if ((round & 1) == 0)
				{
					t_incremental[round] = measure_incremental();
					t_recalculate[round] = measure_recalculate();
				}
				else
				{
					t_recalculate[round] = measure_recalculate();
					t_incremental[round] = measure_incremental();
				}

			double inc = best(t_incremental), full = best(t_recalculate);
			Trace("incremental cost %2d^3 (%d edits): UpdateSolids %.1f us | SetSolid + Recalculate %.1f us | ratio %.3f",
				size, cNumEdits, inc, full, full > 0.0? inc / full : 0.0);

			if (size >= 64)
				CHECK(inc < full);							// the whole point of the incremental path
		}
	}

	// ---- Per-voxel material data must be **self owned** by the shape ----
	// This is the regression test for the Unity editor crash: the shape used to store raw
	// pointers into the caller's managed arrays. The caller (VxlMaterialStore) reallocates or
	// reshuffles those arrays on grow / compact, and material lookups happen inside solver jobs
	// (multi threaded), so borrowing the pointer is a use-after-free waiting to happen.
	// The test scribbles over the source buffers right after SetMaterialData: if the shape
	// still answers correctly afterwards, it really did copy the data.
	TEST_CASE("VoxelShapeMaterialDataIsSelfOwned")
	{
		VoxelObject obj(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCube(obj);
		VoxelShape *shape = const_cast<VoxelShape *>(static_cast<const VoxelShape *>(obj.mShape.GetPtr()));

		// 16x16x16 voxels = 2x2x2 chunks of 8x8x8. One material per chunk => code 0
		// (no number array at all, the record is just the 1 entry index table).
		// header layout: bits 0..19 = base word, bits 20..22 = code, bits 23.. = material count.
		const int cChunks = 8;
		uint32 header[cChunks];
		uint64 data[cChunks];
		for (int i = 0; i < cChunks; ++i)
		{
			header[i] = uint32(i) | (0u << 20) | (1u << 23);	// base = i, code 0, count 1
			data[i] = uint64(10 + i);							// material index of the whole chunk
		}
		shape->SetMaterialData(header, cChunks, data, cChunks);
		CHECK(shape->HasMaterialData());

		// Scribble over the sources: a borrowing shape would now return garbage
		for (int i = 0; i < cChunks; ++i)
		{
			header[i] = 0xFFFFFFFFu;
			data[i] = 0xFFFFFFFFFFFFFFFFull;
		}

		for (int i = 0; i < cChunks; ++i)
		{
			// chunk index = (x>>3) + (y>>3)*2 + (z>>3)*4, pick a voxel well inside that chunk
			int x = (i & 1) * 8 + 3, y = ((i >> 1) & 1) * 8 + 5, z = ((i >> 2) & 1) * 8 + 7;
			CHECK(shape->GetMaterialId(x, y, z) == uint(10 + i));
		}

		// Out of range coordinates / a header pointing outside the pool must yield "no material"
		// instead of reading out of bounds.
		CHECK(shape->GetMaterialId(-1, 0, 0) == 0u);
		CHECK(shape->GetMaterialId(cNumVoxels, 0, 0) == 0u);
		header[0] = 1000000u | (0u << 20) | (1u << 23);			// base way past the pool
		shape->SetMaterialData(header, cChunks, data, cChunks);
		CHECK(shape->GetMaterialId(3, 5, 7) == 0u);

		// Clearing must go back to "no per voxel material"
		shape->SetMaterialData(nullptr, 0, nullptr, 0);
		CHECK(!shape->HasMaterialData());
		CHECK(shape->GetMaterialId(3, 5, 7) == 0u);
	}

	// ---- Per voxel materials must survive AddRef / Release ----
	// Jolt materials derive from RefTarget (reference counted) and the result of GetMaterial() is
	// held in a RefConst<> by upper layers. The character controller does exactly that:
	//     mGroundMaterial = best_contact->mMaterial;   // CharacterVirtual::UpdateSupportingContact
	// The per voxel materials live in a static array that was never allocated with new, so their
	// refcount has to be marked "embedded". Without it the first Release() sees the count go 1 -> 0
	// and runs `delete this;` i.e. operator delete on a static object -> free() of a non heap
	// address -> ntdll reports STATUS_HEAP_CORRUPTION (0xC0000374). That is the crash this test
	// guards: with the embedding missing the suite dies right here, exactly like the editor did
	// while the character walked on voxel terrain.
	TEST_CASE("VoxelShapePerVoxelMaterialSurvivesRefConst")
	{
		VoxelObject obj(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCube(obj);
		VoxelShape *shape = const_cast<VoxelShape *>(static_cast<const VoxelShape *>(obj.mShape.GetPtr()));

		// One material per 8x8x8 chunk => code 0: there is no number array at all, the record is
		// just the 1 entry index table holding the material index of the whole chunk.
		const int cChunks = 8;
		uint32 header[cChunks];
		uint64 data[cChunks];
		for (int i = 0; i < cChunks; ++i)
		{
			header[i] = uint32(i) | (0u << 20) | (1u << 23);	// base = i, code 0, count 1
			data[i] = uint64(i + 1);							// material index 1..8 (0 would mean "no material")
		}
		shape->SetMaterialData(header, cChunks, data, cChunks);

		for (int i = 0; i < cChunks; ++i)
		{
			// A voxel well inside chunk i (chunk index = (x >> 3) + (y >> 3) * 2 + (z >> 3) * 4)
			uint x = uint(i & 1) * 8 + 3, y = uint((i >> 1) & 1) * 8 + 5, z = uint((i >> 2) & 1) * 8 + 7;
			SubShapeID sub_id = shape->EncodeSubShapeID(SubShapeIDCreator(), x, y, z).GetID();

			const PhysicsMaterial *material = shape->GetMaterial(sub_id);
			CHECK(material != nullptr);
			CHECK(material == VoxelShape::sGetVoxelMaterial(uint(i + 1)));

			// Embedded: the counter carries the sentinel, so Release() can never reach the
			// "delete this" branch. A plain static object would sit at 0 here.
			uint32 count = material->GetRefCount();
			CHECK(count > 1);

			// Exactly what the character controller does every time the supporting contact changes
			{
				RefConst<PhysicsMaterial> ref(material);	// AddRef
				CHECK(material->GetRefCount() == count + 1);
			}													// Release
			// The very same object must still be alive and readable (with the bug it was freed above)
			CHECK(material->GetRefCount() == count);
			CHECK(shape->GetMaterial(sub_id) == material);
			CHECK(VoxelShape::sGetVoxelMaterial(uint(i + 1)) == material);
		}

		shape->SetMaterialData(nullptr, 0, nullptr, 0);
		CHECK(!shape->HasMaterialData());
	}

	// ---- The global material table is copy on write, and publishing an identical table is a no-op ----
	// Voxels are edited all the time, and the natural thing for the editor side to do is to re-export
	// the whole table after every edit (VxlMaterialTable.ExportPhysics). Those re-exports almost always
	// contain the same numbers. Old tables are deliberately never freed (a thread may still hold a
	// physics material pointer into one), so without the "identical content" short circuit every single
	// voxel edit would leak another 6 KB table. The short circuit also keeps the pointers stable.
	TEST_CASE("VoxelShapeMaterialTableCopyOnWrite")
	{
		constexpr int cCount = 4;
		float friction[cCount] = { 0.1f, 0.2f, 0.3f, 0.4f };
		float restitution[cCount] = { 0.5f, 0.6f, 0.7f, 0.8f };

		// Remember the current contents so the test leaves the global table the way it found it
		float old_friction[cCount], old_restitution[cCount];
		for (int i = 0; i < cCount; ++i)
		{
			old_friction[i] = VoxelShape::sGetVoxelMaterial(uint(i))->mFriction;
			old_restitution[i] = VoxelShape::sGetVoxelMaterial(uint(i))->mRestitution;
		}
		// An entry beyond inCount must be carried over from the previous table
		float carried = VoxelShape::sGetVoxelMaterial(uint(cCount + 1))->mFriction;

		VoxelShape::sSetMaterialTable(friction, restitution, cCount);
		const VoxelMaterial *published = VoxelShape::sGetVoxelMaterial(0);
		CHECK(published->mFriction == friction[0]);
		CHECK(published->mRestitution == restitution[0]);
		CHECK(VoxelShape::sGetVoxelMaterial(uint(cCount - 1))->mFriction == friction[cCount - 1]);
		CHECK(VoxelShape::sGetVoxelMaterial(uint(cCount + 1))->mFriction == carried);

		// Same content again: nothing is published and the pointer must stay put
		VoxelShape::sSetMaterialTable(friction, restitution, cCount);
		CHECK(VoxelShape::sGetVoxelMaterial(0) == published);

		// A real change has to be published (and visible through the same accessor)
		friction[1] = 0.9f;
		VoxelShape::sSetMaterialTable(friction, restitution, cCount);
		CHECK(VoxelShape::sGetVoxelMaterial(0) != published);
		CHECK(VoxelShape::sGetVoxelMaterial(1)->mFriction == 0.9f);

		// The published materials are embedded as well: a RefConst on a copy-on-write table must not
		// delete it (see VoxelShapePerVoxelMaterialSurvivesRefConst for what happens otherwise)
		{
			RefConst<PhysicsMaterial> ref(VoxelShape::sGetVoxelMaterial(1));
			CHECK(ref->GetRefCount() > 1);
		}
		CHECK(VoxelShape::sGetVoxelMaterial(1)->mFriction == 0.9f);

		// Restore
		VoxelShape::sSetMaterialTable(old_friction, old_restitution, cCount);
		for (int i = 0; i < cCount; ++i)
			CHECK(VoxelShape::sGetVoxelMaterial(uint(i))->mFriction == old_friction[i]);
	}

	TEST_CASE("VoxelShapeTerrainCraterDebris")
	{
		// ★ Unity 里"动态体素 × 地形"的真实几何：
		//   - 地形区块 = 128³ 网格（8 m），**绝大部分是空气**（地表以上整段都是空）；
		//   - 碎块是从地表挖出来的 ⇒ 它停在**坑里**，坑底比周围地表低 0.75 m；
		//
		// mode 0 / mode 3 非边缘分支用的 axis_world = 两个**整网格** OBB 的最小分离轴。
		// 重叠量 = halfA + halfB - |Δcenter·轴|；地形 A 的 halfB = 4 m 远大于碎块的 0.25 m
		// ⇒ "最小重叠轴" 实际就是 **Δcenter 最大的那根轴**，也就是"碎块离区块中心最远的方向"。
		// 碎块在区块正中时 Δy 最大 ⇒ 拿到竖直轴（对）；靠近区块**边缘**时 Δx / Δz 最大
		// ⇒ 拿到纯水平轴 ⇒ 竖直支撑条数为 0，碎块被沿 x/z 推来推去（缝上左右横跳）。
		constexpr int   cSize = 128;
		constexpr float cVs = 0.0625f;
		constexpr int   cSurface = 100;			// 地表：格 0..99 实心，顶面 y = 100*vs - 4
		constexpr int   cFloor = 88;			// 坑底：格 0..87 实心，顶面 y = 88*vs - 4
		const float half = 0.5f * float(cSize) * cVs;	// 网格半边长（4 m）

		VoxelObject cube(8, 8, 8, cVs);
		sMakeCube(cube);

		// 坑在区块正中（ox = 56） vs 坑贴着区块 -x 边缘（ox = 4）
		for (int ox : { 56, 4 })
		{
			VoxelObject terrain(cSize, cSize, cSize, cVs);
			for (int z = 0; z < cSize; ++z)
				for (int y = 0; y < cSurface; ++y)
					for (int x = 0; x < cSize; ++x)
						terrain.Cell(x, y, z) = 1;
			for (int z = 56; z < 66; ++z)
				for (int y = cFloor; y < cSurface; ++y)
					for (int x = ox; x < ox + 10; ++x)
						terrain.Cell(x, y, z) = 0;
			terrain.Update();

			// 跟 Unity 里一样：地形区块在 ±X / ±Z 方向上是延续到隔壁区块的
			const_cast<VoxelShape *>(static_cast<const VoxelShape *>(terrain.mShape.GetPtr()))->SetBorderContinues(true);

			const float cx = (float(ox) + 5.0f) * cVs - half;	// 碎块占格 ox+1 .. ox+8，中心
			const float cz = 61.0f * cVs - half;
			const float rest_y = float(cFloor) * cVs - half + 0.5f * 8.0f * cVs;

			// ---- 1) 静置姿态的接触集合（纯窄相位，不开仿真）----
			{
				CollideShapeSettings settings;
				AllHitCollisionCollector<CollideShapeCollector> collector;
				sCollide(cube.mShape, Mat44::sTranslation(Vec3(cx, rest_y - 0.005f, cz)),
					terrain.mShape, Mat44::sTranslation(Vec3::sZero()), settings, collector);

				int vertical = 0, horizontal = 0;
				float sum_hor = 0.0f;
				for (const CollideShapeResult &h : collector.mHits)
				{
					float ny = abs(h.mPenetrationAxis.GetY());
					sum_hor += 1.0f - ny;
					if (ny > 0.9f) ++vertical; else ++horizontal;
				}
				Trace("坑 ox=%d：接触 %d 条（竖直 %d / 水平 %d）平均水平分量 %.3f｜碎块 (%.3f, %.3f, %.3f)",
					ox, int(collector.mHits.size()), vertical, horizontal,
					collector.mHits.empty()? 0.0f : sum_hor / float(collector.mHits.size()),
					cx, rest_y, cz);

				// ★ 坑在区块正中还是贴着区块边缘，接触集合必须**完全一样**（全是竖直支撑）。
				//  退化时（整网格 SAT 轴被大网格主导）这里会变成"竖直 28 / 水平 36"。
				CAPTURE(ox);
				CHECK(collector.mHits.size() == 64);
				CHECK(vertical == 64);
				CHECK(horizontal == 0);
			}

			// ---- 2) 真跑一遍 ----
			{
				PhysicsTestContext c;
				c.CreateBody(BodyCreationSettings(terrain.mShape, RVec3::sZero(), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);
				Body &body = c.CreateBody(BodyCreationSettings(cube.mShape, RVec3(cx, rest_y + 0.4f, cz), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

				int flips_x = 0, flips_z = 0;
				float prev_vx = 0.0f, prev_vz = 0.0f;
				for (int i = 0; i < 180; ++i)
				{
					c.Simulate(1.0f / 60.0f);
					Vec3 v = body.GetLinearVelocity();
					if (v.GetX() * prev_vx < 0.0f && abs(v.GetX()) > 0.02f) ++flips_x;
					if (v.GetZ() * prev_vz < 0.0f && abs(v.GetZ()) > 0.02f) ++flips_z;
					prev_vx = v.GetX();
					prev_vz = v.GetZ();
				}

				RVec3 p = body.GetPosition();
				float dxf = float(p.GetX()) - cx, dzf = float(p.GetZ()) - cz;
				Trace("坑 ox=%d：仿真 3 s 终点 (%.4f, %.4f, %.4f)，期望 (%.4f, %.4f, %.4f)｜水平漂移 %.4f m｜vx 反向 %d 次 vz 反向 %d 次",
					ox, float(p.GetX()), float(p.GetY()), float(p.GetZ()), cx, rest_y, cz,
					sqrt(dxf * dxf + dzf * dzf), flips_x, flips_z);

				// 停在坑底、不横向漂移（穿透 slop 0.02 之内）、不来回反向
				CAPTURE(ox);
				CHECK(sqrt(dxf * dxf + dzf * dzf) < 0.01f);
				CHECK(abs(float(p.GetY()) - rest_y) < 0.02f);
				CHECK(flips_x == 0);
				CHECK(flips_z == 0);
			}
		}
	}


	TEST_CASE("VoxelShapeTerrainLocalClassify")
	{
		// sNormalMode == 4（局部 2x2x2 邻域分类）与 mode 3 的同场景对照。
		// 几何与 VoxelShapeTerrainCraterDebris 完全相同：坑贴区块 -x 边缘（ox = 4），碎块落在坑里 ——
		// 这正是"整网格 SAT 的全局轴退化"最容易发作的位置。
		// 四种组合 = {mode 3, mode 4} x {BorderContinues 关, 开}：
		//   mode 3 + 关 = 历史 bug（全局轴被 8 m 盒子的侧壁主导，法线变水平）
		//   mode 4 + 关 = 本方案（不依赖"侧壁是假边界"这个外部标记）
		constexpr int   cSize = 128;
		constexpr float cVs = 0.0625f;
		constexpr int   cSurface = 100;
		constexpr int   cFloor = 88;
		constexpr int   ox = 4;
		const float half = 0.5f * float(cSize) * cVs;

		VoxelObject cube(8, 8, 8, cVs);
		sMakeCube(cube);

		VoxelObject terrain(cSize, cSize, cSize, cVs);
		for (int z = 0; z < cSize; ++z)
			for (int y = 0; y < cSurface; ++y)
				for (int x = 0; x < cSize; ++x)
					terrain.Cell(x, y, z) = 1;
		for (int z = 56; z < 66; ++z)
			for (int y = cFloor; y < cSurface; ++y)
				for (int x = ox; x < ox + 10; ++x)
					terrain.Cell(x, y, z) = 0;
		terrain.Update();

		const float cx = (float(ox) + 5.0f) * cVs - half;
		const float cz = 61.0f * cVs - half;
		const float rest_y = float(cFloor) * cVs - half + 0.5f * 8.0f * cVs;

		VoxelShape *terrain_shape = const_cast<VoxelShape *>(static_cast<const VoxelShape *>(terrain.mShape.GetPtr()));

		for (int border = 0; border < 2; ++border)
			for (int mode : { 3, 4 })
			{
				terrain_shape->SetBorderContinues(border != 0);
				NormalModeScope scope(mode);

				// ---- 1) 静置姿态的接触集合（纯窄相位）----
				{
					CollideShapeSettings settings;
					AllHitCollisionCollector<CollideShapeCollector> collector;
					sCollide(cube.mShape, Mat44::sTranslation(Vec3(cx, rest_y - 0.005f, cz)),
						terrain.mShape, Mat44::sTranslation(Vec3::sZero()), settings, collector);

					int vertical = 0, horizontal = 0;
					float sum_hor = 0.0f;
					for (const CollideShapeResult &h : collector.mHits)
					{
						float ny = abs(h.mPenetrationAxis.GetY());
						sum_hor += 1.0f - ny;
						if (ny > 0.9f) ++vertical; else ++horizontal;
					}
					Trace("mode %d 掩码 %d：接触 %d 条（竖直 %d / 水平 %d）平均水平分量 %.3f",
						mode, border, int(collector.mHits.size()), vertical, horizontal,
						collector.mHits.empty()? 0.0f : sum_hor / float(collector.mHits.size()));

					// mode 3 + 掩码关 = 已知的退化组合（保留它作为"修好了"的判据），只记录不断言
					if (mode == 4 || border != 0)
					{
						CAPTURE(mode);
						CAPTURE(border);
						CHECK(vertical > 0);
						CHECK(horizontal == 0);
					}
				}

				// ---- 2) 真跑一遍 ----
				{
					PhysicsTestContext c;
					c.CreateBody(BodyCreationSettings(terrain.mShape, RVec3::sZero(), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);
					Body &body = c.CreateBody(BodyCreationSettings(cube.mShape, RVec3(cx, rest_y + 0.4f, cz), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

					int flips_x = 0, flips_z = 0;
					float prev_vx = 0.0f, prev_vz = 0.0f;
					for (int i = 0; i < 180; ++i)
					{
						c.Simulate(1.0f / 60.0f);
						Vec3 v = body.GetLinearVelocity();
						if (v.GetX() * prev_vx < 0.0f && abs(v.GetX()) > 0.02f) ++flips_x;
						if (v.GetZ() * prev_vz < 0.0f && abs(v.GetZ()) > 0.02f) ++flips_z;
						prev_vx = v.GetX();
						prev_vz = v.GetZ();
					}

					RVec3 p = body.GetPosition();
					float dxf = float(p.GetX()) - cx, dzf = float(p.GetZ()) - cz;
					Trace("mode %d 掩码 %d：仿真 3 s 终点 (%.4f, %.4f, %.4f)｜水平漂移 %.4f m｜vx 反向 %d 次 vz 反向 %d 次",
						mode, border, float(p.GetX()), float(p.GetY()), float(p.GetZ()),
						sqrt(dxf * dxf + dzf * dzf), flips_x, flips_z);

					if (mode == 4 || border != 0)
					{
						CAPTURE(mode);
						CAPTURE(border);
						CHECK(sqrt(dxf * dxf + dzf * dzf) < 0.01f);
						CHECK(abs(float(p.GetY()) - rest_y) < 0.02f);
						CHECK(flips_x == 0);
						CHECK(flips_z == 0);
					}
				}
			}

		terrain_shape->SetBorderContinues(false);
	}



	TEST_CASE("VoxelShapeLocalClassifyOverlap")
	{
		// 两个 8^3 实心方块、半格（0.25 m）错位重叠 —— "两个体素碰撞体重叠推不开"的最小场景。
		// mode 4 在"8 格全实心"时退回整网格全局轴 ⇒ 期望与 mode 0 同量级（推开、不横飞）。
		VoxelObject box_a(8, 8, 8, 0.0625f);
		VoxelObject box_b(8, 8, 8, 0.0625f);
		sMakeCube(box_a);
		sMakeCube(box_b);

		const RVec3 pos_a(0.0f, 0.5f, 0.0f);
		const RVec3 pos_b(0.25f, 0.75f, 0.0f);			// 水平、垂直各重叠 0.25 m

		for (int mode : { 0, 3, 4 })
		{
			NormalModeScope scope(mode);

			// 1) 接触法线分布
			{
				CollideShapeSettings settings;
				AllHitCollisionCollector<CollideShapeCollector> collector;
				sCollide(box_a.mShape, Mat44::sTranslation(pos_a), box_b.mShape, Mat44::sTranslation(pos_b), settings, collector);

				int up = 0, horizontal = 0;
				for (const CollideShapeResult &h : collector.mHits)
				{
					if (abs(h.mPenetrationAxis.GetY()) > 0.9f) ++up; else ++horizontal;
				}
				Trace("方块重叠 mode %d：接触 %d 条（竖直 %d / 水平 %d）", mode, int(collector.mHits.size()), up, horizontal);
			}

			// 2) 真跑：两个动态方块应该被推开（中心距 >= 0.5 m 才算贴合）
			{
				PhysicsTestContext c;
				c.GetSystem()->SetGravity(Vec3::sZero());
				Body &ba = c.CreateBody(BodyCreationSettings(box_a.mShape, pos_a, Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);
				Body &bb = c.CreateBody(BodyCreationSettings(box_b.mShape, pos_b, Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);
				for (int i = 0; i < 120; ++i)
					c.Simulate(1.0f / 60.0f);

				RVec3 pa = ba.GetPosition(), pb = bb.GetPosition();
				Vec3 d = Vec3(pb - pa);
				Trace("方块重叠 mode %d：2 s 后中心距 %.4f m｜a 位移 (%.3f, %.3f, %.3f) b 位移 (%.3f, %.3f, %.3f)",
					mode, d.Length(),
					float(pa.GetX() - pos_a.GetX()), float(pa.GetY() - pos_a.GetY()), float(pa.GetZ() - pos_a.GetZ()),
					float(pb.GetX() - pos_b.GetX()), float(pb.GetY() - pos_b.GetY()), float(pb.GetZ() - pos_b.GetZ()));

				CHECK(d.Length() > 0.45f);
			}
		}
	}


	/// 性能对照：mode 2 / 3 / 4 在**同场景、同格对集合**下的窄相位耗时（µs / query）。
	///
	/// 三个场景对应实际游戏里会遇到的三种接触：
	///   A 平地表（落体静置）：两侧都是平面格（各外露 1 面）=> mode 3 直接保持全局轴，0 额外成本；
	///     mode 4 是它最吃亏的场合（每对都要数 8 个格）。
	///   B 坑壁 / 台阶（挖矿、碎块压在棱上）：棱角格多 => mode 3 走格心连线（1 次 sqrt），
	///     mode 4 走局部 2x2x2 分类（8 次 IsSolid + 若干 sqrt）。
	///   C 两个 8^3 方块斜向重叠：体素 vs 体素，两边都是小网格（无 8 m 空气盒的问题），
	///     是这三个模式共同的历史基准。
	///
	/// 计时方式：warmup 一次后跑 cRounds 轮、每轮 cNumQueries 次**纯窄相位**调用
	/// （sCollideShapeVsShape，collector 只计数、不存 hit、不分配），取最小轮与中位轮。
	/// 最小轮最接近"无干扰的纯计算"，中位轮用来判断抖动。
	TEST_CASE("VoxelShapeNormalModePerformance")
	{
		constexpr int cNumQueries = 200;
		constexpr int cRounds = 5;

		struct Timing
		{
			double			mBest = 0.0;
			double			mMedian = 0.0;
			int				mHits = 0;
		};

		auto sTime = [&](auto &inQuery) -> Timing
		{
			Timing result;
			{
				CountingCollideShapeCollector warm;
				inQuery(warm);
				result.mHits = warm.mCount;
			}

			double samples[8];				// cRounds <= 8（MSVC 不允许在 lambda 里用 constexpr 变量做数组维度）
			for (int r = 0; r < cRounds; ++r)
			{
				chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
				for (int i = 0; i < cNumQueries; ++i)
				{
					CountingCollideShapeCollector counting;
					inQuery(counting);
				}
				samples[r] = double(chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - start).count()) / (1000.0 * cNumQueries);
			}

			// 插入排序（5 个元素，不值得引入 <algorithm>）
			for (int i = 1; i < cRounds; ++i)
			{
				double v = samples[i];
				int j = i - 1;
				while (j >= 0 && samples[j] > v)
				{
					samples[j + 1] = samples[j];
					--j;
				}
				samples[j + 1] = v;
			}
			result.mBest = samples[0];
			result.mMedian = samples[cRounds / 2];
			return result;
		};

		// ---- 场景 A / B 共用：128^3 地形（8 m 盒子），坑贴 -x 边缘 ----
		constexpr int   cSize = 128;
		constexpr int   cSurface = 100;
		constexpr int   cFloor = 88;
		constexpr int   ox = 4;
		const float half = 0.5f * float(cSize) * cVoxelSize;

		VoxelObject terrain_flat(cSize, cSize, cSize, cVoxelSize);
		for (int z = 0; z < cSize; ++z)
			for (int y = 0; y < cSurface; ++y)
				for (int x = 0; x < cSize; ++x)
					terrain_flat.Cell(x, y, z) = 1;
		terrain_flat.Update();

		VoxelObject terrain_pit(cSize, cSize, cSize, cVoxelSize);
		for (int z = 0; z < cSize; ++z)
			for (int y = 0; y < cSurface; ++y)
				for (int x = 0; x < cSize; ++x)
					terrain_pit.Cell(x, y, z) = 1;
		for (int z = 56; z < 66; ++z)
			for (int y = cFloor; y < cSurface; ++y)
				for (int x = ox; x < ox + 10; ++x)
					terrain_pit.Cell(x, y, z) = 0;
		terrain_pit.Update();

		VoxelObject cube(8, 8, 8, cVoxelSize);
		sMakeCube(cube);

		CollideShapeSettings settings;

		// ---- 场景 A：平地表，碎块落在区块正上方（远离侧壁）----
		{
			const float ax = 64.0f * cVoxelSize - half;
			const float az = 64.0f * cVoxelSize - half;
			const float ay = float(cSurface) * cVoxelSize - half + 0.25f - 0.005f;

			for (int mode : { 1, 2, 3, 4 })
			{
				NormalModeScope scope(mode);
				auto query = [&](CollideShapeCollector &ioCollector)
				{
					CollisionDispatch::sCollideShapeVsShape(cube.mShape, terrain_flat.mShape, Vec3::sOne(), Vec3::sOne(), Mat44::sTranslation(Vec3(ax, ay, az)), Mat44::sTranslation(Vec3::sZero()), SubShapeIDCreator(), SubShapeIDCreator(), settings, ioCollector);
				};
				Timing t = sTime(query);
				Trace("A flat surface  mode %d：%7.3f us / query（中位 %7.3f，%d hits）", mode, t.mBest, t.mMedian, t.mHits);
			}
		}

		// ---- 场景 B：坑壁，碎块落在坑底靠 -x 侧壁 ----
		{
			const float bx = (float(ox) + 5.0f) * cVoxelSize - half;
			const float bz = 61.0f * cVoxelSize - half;
			const float by = float(cFloor) * cVoxelSize - half + 0.25f - 0.005f;

			for (int mode : { 1, 2, 3, 4 })
			{
				NormalModeScope scope(mode);
				auto query = [&](CollideShapeCollector &ioCollector)
				{
					CollisionDispatch::sCollideShapeVsShape(cube.mShape, terrain_pit.mShape, Vec3::sOne(), Vec3::sOne(), Mat44::sTranslation(Vec3(bx, by, bz)), Mat44::sTranslation(Vec3::sZero()), SubShapeIDCreator(), SubShapeIDCreator(), settings, ioCollector);
				};
				Timing t = sTime(query);
				Trace("B pit wall      mode %d：%7.3f us / query（中位 %7.3f，%d hits）", mode, t.mBest, t.mMedian, t.mHits);
			}
		}

		// ---- 场景 C：两个 8^3 方块，斜 30 度、重叠 0.1 m ----
		{
			VoxelObject box_a(8, 8, 8, cVoxelSize);
			VoxelObject box_b(8, 8, 8, cVoxelSize);
			sMakeCube(box_a);
			sMakeCube(box_b);

			Mat44 t_a = Mat44::sTranslation(Vec3(0, 0.25f, 0));
			Mat44 t_b = Mat44::sRotationTranslation(Quat::sRotation(Vec3::sAxisY(), DegreesToRadians(30.0f)), Vec3(0.15f, 0.35f, 0.0f));

			for (int mode : { 1, 2, 3, 4 })
			{
				NormalModeScope scope(mode);
				auto query = [&](CollideShapeCollector &ioCollector)
				{
					CollisionDispatch::sCollideShapeVsShape(box_a.mShape, box_b.mShape, Vec3::sOne(), Vec3::sOne(), t_a, t_b, SubShapeIDCreator(), SubShapeIDCreator(), settings, ioCollector);
				};
				Timing t = sTime(query);
				Trace("C block overlap mode %d：%7.3f us / query（中位 %7.3f，%d hits）", mode, t.mBest, t.mMedian, t.mHits);
			}
		}

		// ---- 场景 D：两个 8^3 方块**浅**重叠 0.02 m（无旋转）----
		// C 是"两块全面叠在一起"的最坏情况（1024 个格对）；D 是真实碎块互相压住的样子
		// （只有接触面那一层，64 个格对），用来判断 C 的差距会不会落在实际游戏里。
		{
			VoxelObject box_a(8, 8, 8, cVoxelSize);
			VoxelObject box_b(8, 8, 8, cVoxelSize);
			sMakeCube(box_a);
			sMakeCube(box_b);

			Mat44 t_a = Mat44::sTranslation(Vec3(0, 0.25f, 0));
			Mat44 t_b = Mat44::sTranslation(Vec3(0.01f, 0.75f - 0.02f, 0));

			for (int mode : { 1, 2, 3, 4 })
			{
				NormalModeScope scope(mode);
				auto query = [&](CollideShapeCollector &ioCollector)
				{
					CollisionDispatch::sCollideShapeVsShape(box_a.mShape, box_b.mShape, Vec3::sOne(), Vec3::sOne(), t_a, t_b, SubShapeIDCreator(), SubShapeIDCreator(), settings, ioCollector);
				};
				Timing t = sTime(query);
				Trace("D shallow touch mode %d：%7.3f us / query（中位 %7.3f，%d hits）", mode, t.mBest, t.mMedian, t.mHits);
			}
		}
	}



	// ---- mode 4 分类/几何的**逐位**回归 ----
	//
	// 改 mode 4 的任何几何（胶囊、空位线段、查表）都必须保持 acc4 / contact_axis **逐位不变**：
	// 1 ulp 的差在深穿透里会被放大成"推力方向不同"。
	//
	// 做法：随机噪声网格 + 96 组姿态，把所有接触的 (SubShapeID, 法线, 深度, 两个接触点)
	// 按 float 位模式累计成 FNV 哈希。改动前后各跑一次比 `hash=` 那一行即可
	// （改动的正确用法是"先记录旧哈希，改完必须一样"）。
	/// ---- 大物体 vs 地形：窄相位成本复现（用户报告：地形场景里出现大物体非常卡）----
	///
	/// 结构上的两个放大器：
	///   1. 遍历范围 = 对方**整网格 OBB 的世界盒**（含空格）：大而空的物体把地形侧圈得很大；
	///      反过来地形整块 OBB（8 m 盒）也把物体侧圈满 —— 靠"遍历 count 小的一侧"兜底。
	///   2. 内部格 x 内部格的"埋没接触"照常遍历（sSkipInteriorPairs 默认关）。
	/// 本用例只量化现状与开关对照，不改引擎行为。
	TEST_CASE("VoxelShapeBigObjectVsTerrain")
	{
		constexpr int cTerrain = 128;
		constexpr int cSurface = 100;
		constexpr int cNumQueries = 100;
		constexpr int cRounds = 5;
		const float half = 0.5f * float(cTerrain) * cVoxelSize;

		VoxelObject terrain(cTerrain, cTerrain, cTerrain, cVoxelSize);
		for (int z = 0; z < cTerrain; ++z)
			for (int y = 0; y < cSurface; ++y)
				for (int x = 0; x < cTerrain; ++x)
					terrain.Cell(x, y, z) = 1;
		terrain.Update();

		// 预热派生统计（mSolidMin/Max）—— 对应 C# 侧建块后的 GetNumSolidVoxels() 预热：
		// 窄相位的 sGetCellRange 靠它把遍历范围夹到实心格内，静态块不预热就退回整块网格。
		((const VoxelShape *)terrain.mShape.GetPtr())->GetNumSolidVoxels();

		CollideShapeSettings settings;
		Mat44 terrain_t = Mat44::sTranslation(Vec3::sZero());

		// 物体下沿相对地表的偏移（世界坐标，地表在 cSurface * cVoxelSize - half）：
		// 正 = 嵌入地表，负 = 悬空。
		const float ground_y = float(cSurface) * cVoxelSize - half;

		VoxelObject solid32(32, 32, 32, cVoxelSize);
		for (int z = 0; z < 32; ++z)
			for (int y = 0; y < 32; ++y)
				for (int x = 0; x < 32; ++x)
					solid32.Cell(x, y, z) = 1;
		solid32.Update();

		VoxelObject shell32(32, 32, 32, cVoxelSize);
		for (int z = 0; z < 32; ++z)
			for (int y = 0; y < 32; ++y)
				for (int x = 0; x < 32; ++x)
				{
					bool shell = x < 3 || y < 3 || z < 3 || x >= 29 || y >= 29 || z >= 29;
					shell32.Cell(x, y, z) = shell? 1 : 0;
				}
		shell32.Update();

		VoxelObject solid64(64, 64, 64, cVoxelSize);
		for (int z = 0; z < 64; ++z)
			for (int y = 0; y < 64; ++y)
				for (int x = 0; x < 64; ++x)
					solid64.Cell(x, y, z) = 1;
		solid64.Update();

		struct Timing { double mBest; double mMedian; int mHits; };
		auto sTime = [&](auto &inQuery) -> Timing
		{
			CountingCollideShapeCollector warm;
			inQuery(warm);
			Timing result;
			result.mBest = 0.0;
			result.mMedian = 0.0;
			result.mHits = warm.mCount;

			double samples[8];
			for (int r = 0; r < cRounds; ++r)
			{
				chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
				for (int i = 0; i < cNumQueries; ++i)
				{
					CountingCollideShapeCollector counting;
					inQuery(counting);
				}
				samples[r] = double(chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - start).count()) / (1000.0 * cNumQueries);
			}
			for (int i = 1; i < cRounds; ++i)
			{
				double v = samples[i];
				int j = i - 1;
				while (j >= 0 && samples[j] > v) { samples[j + 1] = samples[j]; --j; }
				samples[j + 1] = v;
			}
			result.mBest = samples[0];
			result.mMedian = samples[cRounds / 2];
			return result;
		};

		struct Scene { const char *mName; const VoxelObject *mObject; float mLift; };
		Scene scenes[] =
		{
			{ "L1 solid32 lift 0.10", &solid32, 0.10f },
			{ "L2 solid32 lift 0.50", &solid32, 0.50f },
			{ "L3 shell32 lift 0.10", &shell32, 0.10f },
			{ "L4 solid64 lift 0.10", &solid64, 0.10f },
			{ "L5 solid32 sink 0.05", &solid32, -0.05f },
			{ "L6 solid32 sink 1.50", &solid32, -1.50f },
			{ "L7 solid32 lift 4.00", &solid32, 4.00f },
		};

		NormalModeScope scope4(4);
		for (const Scene &sc : scenes)
		{
			float obj_half = 0.5f * float(sc.mObject->mSizeX) * cVoxelSize;
			Mat44 obj_t = Mat44::sTranslation(Vec3(0.0f, ground_y + obj_half - sc.mLift, 0.0f));
			for (int skip = 0; skip < 2; ++skip)
			{
				VoxelShape::sInteriorSkipPairs = skip;			// 0 = off, 1 = both-interior（2 = either 不在本对照里）
				auto query = [&](CollideShapeCollector &ioCollector)
				{
					CollisionDispatch::sCollideShapeVsShape(sc.mObject->mShape, terrain.mShape, Vec3::sOne(), Vec3::sOne(), obj_t, terrain_t, SubShapeIDCreator(), SubShapeIDCreator(), settings, ioCollector);
				};
				Timing t = sTime(query);
				Trace("%-22s skipInterior=%d：%10.1f us / query（中位 %10.1f，%d hits）", sc.mName, skip, t.mBest, t.mMedian, t.mHits);
			}
		}
		VoxelShape::sInteriorSkipPairs = 0;
	}

	/// ---- 5m 大球（80^3 内切球）落在 128^3 地形上：复现"落地后越来越卡"的窄相位成本 ----
	/// 场景 S1~S4：穿透 0.5 / 2 / 5 格（静止~深）+ 悬空 1 格。对照遍历侧选择 / chunk 剪枝的效果。
	TEST_CASE("VoxelShapeSphereVsTerrainPerf")
	{
		constexpr int cTerrain = 128;
		constexpr int cSurface = 100;
		constexpr int cBall = 80;
		constexpr float cBallR = 40.0f;			// 内切球半径（格）
		constexpr int cNumQueries = 50;
		constexpr int cRounds = 5;
		const float half = 0.5f * float(cTerrain) * cVoxelSize;

		VoxelObject terrain(cTerrain, cTerrain, cTerrain, cVoxelSize);
		for (int z = 0; z < cTerrain; ++z)
			for (int y = 0; y < cSurface; ++y)
				for (int x = 0; x < cTerrain; ++x)
					terrain.Cell(x, y, z) = 1;
		terrain.Update();
		// 预热派生统计（对应 C# 建块后的 GetNumSolidVoxels() ；不然实心 AABB 回退整网格，meet 盒高达 28 层）
		((const VoxelShape *)terrain.mShape.GetPtr())->GetNumSolidVoxels();

		VoxelObject ball(cBall, cBall, cBall, cVoxelSize);
		{
			const float h = 0.5f * float(cBall);
			for (int z = 0; z < cBall; ++z)
				for (int y = 0; y < cBall; ++y)
					for (int x = 0; x < cBall; ++x)
					{
						float cx = (x + 0.5f) - h, cy = (y + 0.5f) - h, cz = (z + 0.5f) - h;
						if (cx * cx + cy * cy + cz * cz <= cBallR * cBallR)
							ball.Cell(x, y, z) = 1;
					}
			ball.Update();
		}

		CollideShapeSettings settings;
		const float cx_w = 64.0f * cVoxelSize - half;
		const float cz_w = cx_w;

		struct Scene { const char *mName; float mPen; };
		Scene scenes[] =
		{
			{ "S1 rest  pen 0.5", 0.5f },
			{ "S2 pen 2 cells  ", 2.0f },
			{ "S3 pen 5 cells  ", 5.0f },
			{ "S4 hover 1 cell ", -1.0f },
		};

		for (const Scene &sc : scenes)
		{
			// 球底世界 y = (cSurface - pen) * vs - half；球心 = 球底 + 半径
			const float cy_w = (float(cSurface) - sc.mPen + cBallR) * cVoxelSize - half;

			for (int mode : { 4, 1 })
			{
				NormalModeScope scope(mode);
				auto query = [&](CollideShapeCollector &ioCollector)
				{
					CollisionDispatch::sCollideShapeVsShape(ball.mShape, terrain.mShape, Vec3::sOne(), Vec3::sOne(),
						Mat44::sTranslation(Vec3(cx_w, cy_w, cz_w)), Mat44::sTranslation(Vec3::sZero()),
						SubShapeIDCreator(), SubShapeIDCreator(), settings, ioCollector);
				};
				CountingCollideShapeCollector warm;
				query(warm);
				double samples[8];
				for (int r = 0; r < cRounds; ++r)
				{
					chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
					for (int i = 0; i < cNumQueries; ++i)
					{
						CountingCollideShapeCollector counting;
						query(counting);
					}
					samples[r] = double(chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - start).count()) / (1000.0 * cNumQueries);
				}
				for (int i = 1; i < cRounds; ++i)
				{
					double v = samples[i]; int j = i - 1;
					while (j >= 0 && samples[j] > v) { samples[j + 1] = samples[j]; --j; }
					samples[j + 1] = v;
				}
				Trace("%s mode %d：%9.1f us / query（中位 %9.1f，%d hits）", sc.mName, mode, samples[0], samples[cRounds / 2], warm.mCount);			}
		}
	}

	/// 完整物理步计时（接触生成 + 求解 + 回调），镜像 Unity 配置：MaxContactConstraints = 4096、单线程、1/60。
	/// 用于区分"引擎窄相位"与"上层每接触回调"的开销归属。
	TEST_CASE("VoxelShapeFullStepPerf")
	{
		constexpr int cTerrain = 128;
		constexpr int cSurface = 100;
		constexpr int cBall = 80;
		constexpr int cBallR = 40;
		const float half = 0.5f * float(cTerrain) * cVoxelSize;

		// (dt, collisionSteps, workerThreads, maxBodies, maxBodyPairs, maxContactConstraints)
		PhysicsTestContext c(1.0f / 60.0f, 1, 0, 1024, 8192, 4096);

		VoxelObject terrain(cTerrain, cTerrain, cTerrain, cVoxelSize);
		for (int z = 0; z < cTerrain; ++z)
			for (int y = 0; y < cSurface; ++y)
				for (int x = 0; x < cTerrain; ++x)
					terrain.Cell(x, y, z) = 1;
		terrain.Update();
		// 预热派生统计（对应 C# 建块后的 GetNumSolidVoxels()；不然实心 AABB 回退整网格，
		// meet 盒厚达 28 层 —— Unity 游戏路径在建块时已预热，无此问题）
		((const VoxelShape *)terrain.mShape.GetPtr())->GetNumSolidVoxels();

		VoxelObject ball(cBall, cBall, cBall, cVoxelSize);
		{
			const float h = 0.5f * float(cBall);
			for (int z = 0; z < cBall; ++z)
				for (int y = 0; y < cBall; ++y)
					for (int x = 0; x < cBall; ++x)
					{
						float cx = (x + 0.5f) - h, cy = (y + 0.5f) - h, cz = (z + 0.5f) - h;
						if (cx * cx + cy * cy + cz * cz <= float(cBallR * cBallR))
							ball.Cell(x, y, z) = 1;
					}
			ball.Update();
		}

		c.CreateBody(BodyCreationSettings(terrain.mShape, RVec3::sZero(), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

		struct CountingListener : public ContactListener
		{
			int						mAdded = 0;
			int						mPersisted = 0;
			int						mPoints = 0;		// 本步所有流形的接触点总数
			int						mMaxPoints = 0;		// 单条流形最大点数

			virtual void			OnContactAdded(const Body &, const Body &, const ContactManifold &inManifold, ContactSettings &) override { Sample(inManifold); }
			virtual void			OnContactPersisted(const Body &, const Body &, const ContactManifold &inManifold, ContactSettings &) override { Sample(inManifold); }
			void					Sample(const ContactManifold &inManifold)
			{
				int pc = int(inManifold.mRelativeContactPointsOn1.size());
				++mAdded; // 与回调条数同口径
				mPoints += pc;
				if (pc > mMaxPoints) mMaxPoints = pc;
			}
		} listener;
		c.GetSystem()->SetContactListener(&listener);

		// 地形表面世界 y = cSurface*vs - half（shape 局部原点在网格中心）；球悬空 2 m 落下
		const float surface_y = float(cSurface) * cVoxelSize - half;
		const float ball_r = float(cBallR) * cVoxelSize;
		BodyID ball_id = c.GetBodyInterface().CreateAndAddBody(BodyCreationSettings(ball.mShape,
			RVec3(0, surface_y + ball_r + 2.0f, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

		// 关休眠：量"贴地稳态"的真实每步开销（休眠后零接触会骗人）
		{
			PhysicsSettings s = c.GetSystem()->GetPhysicsSettings();
			s.mAllowSleeping = false;
			c.GetSystem()->SetPhysicsSettings(s);
		}

		c.Simulate(2.0f);	// 落地并进入稳定贴地
		Trace("落地后球心 y = %f（surface %f，活动体数 %d）",
			double(c.GetBodyInterface().GetPosition(ball_id).GetY()), double(surface_y), c.GetSystem()->GetNumActiveBodies(EBodyType::RigidBody));

		constexpr int cSteps = 120;
		double samples[cSteps];
		double contacts[cSteps];
		double points[cSteps];
		double qu[cSteps];
		int max_points = 0;
		for (int i = 0; i < cSteps; ++i)
		{
			listener.mAdded = 0;
			listener.mPoints = 0;
			chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
			c.SimulateSingleStep();
			samples[i] = double(chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - start).count()) / 1000.0;
			contacts[i] = double(listener.mAdded);
			points[i] = double(listener.mPoints);
			if (listener.mMaxPoints > max_points) max_points = listener.mMaxPoints;
		}
		for (int i = 1; i < cSteps; ++i)
		{
			double v = samples[i]; int j = i - 1;
			while (j >= 0 && samples[j] > v) { samples[j + 1] = samples[j]; --j; }
			samples[j + 1] = v;
			double w = contacts[i]; j = i - 1;
			while (j >= 0 && contacts[j] > w) { contacts[j + 1] = contacts[j]; --j; }
			contacts[j + 1] = w;
			double u = points[i]; j = i - 1;
			while (j >= 0 && points[j] > u) { points[j + 1] = points[j]; --j; }
			points[j + 1] = u;
		}
		double sum = 0.0, csum = 0.0, psum = 0.0;
		for (int i = 0; i < cSteps; ++i) { sum += samples[i]; csum += contacts[i]; psum += points[i]; }
		Trace("FullStep 120 步：中位 %9.1f us / 步（平均 %9.1f，最大 %9.1f）；每步流形回调中位 %6.0f 条（平均 %6.1f）；每步接触点中位 %6.0f（平均 %6.1f，单流形最大 %d）",
			samples[cSteps / 2], sum / cSteps, samples[cSteps - 1], contacts[cSteps / 2], csum / cSteps, points[cSteps / 2], psum / cSteps, max_points);
		CHECK(c.GetBodyInterface().GetPosition(ball_id).GetY() > surface_y);	// 球应停在地形表面上方

		// B 段：从 10 m 高砸落，量落地瞬间的深贯穿步（球会带上翻滚旋转，走旋转网格快路径）；
		// 独立查询一列用于归属：整步耗时 ≈ 窄相位查询 + 求解/回调的固定开销
		c.GetBodyInterface().SetPosition(ball_id, RVec3(0, surface_y + ball_r + 10.0f, 0), EActivation::Activate);
		c.Simulate(1.30f);	// 自由落体 10 m 约 1.43 s，量测窗正落在刚着地的深贯穿段
		max_points = 0;
		for (int i = 0; i < cSteps; ++i)
		{
			listener.mAdded = 0;
			listener.mPoints = 0;
			chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
			c.SimulateSingleStep();
			samples[i] = double(chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - start).count()) / 1000.0;
			// 临时对照：同姿态独立窄相位查询（160^3 球，物理系统的 max_sep 设置）
			RVec3 bpos = c.GetBodyInterface().GetPosition(ball_id);
			Quat brot = c.GetBodyInterface().GetRotation(ball_id);
			chrono::high_resolution_clock::time_point q0 = chrono::high_resolution_clock::now();
			{
				CountingCollideShapeCollector qc;
				CollideShapeSettings cs2;
				cs2.mMaxSeparationDistance = 0.02f;
				CollisionDispatch::sCollideShapeVsShape(ball.mShape, terrain.mShape, Vec3::sOne(), Vec3::sOne(),
					Mat44::sRotationTranslation(brot, Vec3(float(bpos.GetX()), float(bpos.GetY()), float(bpos.GetZ()))),
					Mat44::sTranslation(Vec3::sZero()), SubShapeIDCreator(), SubShapeIDCreator(), cs2, qc);
			}
			qu[i] = double(chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - q0).count()) / 1000.0;
			contacts[i] = double(listener.mAdded);
			points[i] = double(listener.mPoints);
			if (listener.mMaxPoints > max_points) max_points = listener.mMaxPoints;
		}
		for (int i = 1; i < cSteps; ++i)
		{
			double v = samples[i]; int j = i - 1;
			while (j >= 0 && samples[j] > v) { samples[j + 1] = samples[j]; --j; }
			samples[j + 1] = v;
			double w = contacts[i]; j = i - 1;
			while (j >= 0 && contacts[j] > w) { contacts[j + 1] = contacts[j]; --j; }
			contacts[j + 1] = w;
			double u = points[i]; j = i - 1;
			while (j >= 0 && points[j] > u) { points[j + 1] = points[j]; --j; }
			points[j + 1] = u;
		}
		sum = csum = psum = 0.0;
		double qsum = 0.0;
		for (int i = 0; i < cSteps; ++i) { sum += samples[i]; csum += contacts[i]; psum += points[i]; qsum += qu[i]; }
		{
			double qv[cSteps];
			memcpy(qv, qu, sizeof(qv));
			for (int i = 1; i < cSteps; ++i) { double v = qv[i]; int j = i - 1; while (j >= 0 && qv[j] > v) { qv[j + 1] = qv[j]; --j; } qv[j + 1] = v; }
			Trace("Impact  120 步：中位 %9.1f us / 步（独立查询中位 %9.1f us，平均 %9.1f us）；每步流形回调中位 %6.0f 条；每步接触点中位 %6.0f（单流形最大 %d）",
				samples[cSteps / 2], qv[cSteps / 2], qsum / cSteps, contacts[cSteps / 2], points[cSteps / 2], max_points);
		}
	}

	/// 把一个立方等边的 VoxelObject 地形按 chunk³ 拆进 MutableCompoundShape（镜像 C# 地形分块方案）：
	/// 每个 chunk 一个子形状，空 chunk 不加 —— broadphase / 窄相位的子形状包围盒过滤天然剔除。
	static Ref<Shape> sBuildChunkedCompound(const VoxelObject &inTerrain, int inChunk, Array<VoxelObject *> &ioStorage)
	{
		const int n = inChunk;
		const int cn = inTerrain.mSizeX / n;
		const float vs = inTerrain.mVoxelSize;
		MutableCompoundShapeSettings settings;
		for (int cz = 0; cz < cn; ++cz)
			for (int cy = 0; cy < cn; ++cy)
				for (int cx = 0; cx < cn; ++cx)
				{
					bool any = false;
					for (int z = 0; z < n && !any; ++z)
						for (int y = 0; y < n && !any; ++y)
							for (int x = 0; x < n && !any; ++x)
								any = inTerrain.Get(cx * n + x, cy * n + y, cz * n + z);
					if (!any)
						continue;	// 空 chunk：不加形状
					VoxelObject *chunk = new VoxelObject(n, n, n, vs);
					ioStorage.push_back(chunk);
					for (int z = 0; z < n; ++z)
						for (int y = 0; y < n; ++y)
							for (int x = 0; x < n; ++x)
								chunk->Cell(x, y, z) = inTerrain.Get(cx * n + x, cy * n + y, cz * n + z) ? 1 : 0;
					chunk->Update();
					static_cast<const VoxelShape *>(chunk->mShape.GetPtr())->GetNumSolidVoxels();	// 预热派生统计
					Vec3 off = Vec3((cx + 0.5f) * n, (cy + 0.5f) * n, (cz + 0.5f) * n) * vs
						- Vec3(0.5f * inTerrain.mSizeX, 0.5f * inTerrain.mSizeY, 0.5f * inTerrain.mSizeZ) * vs;
					settings.AddShape(off, Quat::sIdentity(), chunk->mShape, 0);
				}
		return settings.Create().Get();
	}

	/// 复合 chunk 地形（C# 分块方案的引擎侧等价物）vs 整块单网格：
	/// 接触集合（数量 + 深度/接触点总和，容差对照——复合路径的变换多乘一次子形状偏移，浮点有 1e-6 级差异）
	/// 必须一致，并量旋转姿态下窄相位查询耗时与完整物理步耗时。
	TEST_CASE("VoxelShapeCompoundTerrain")
	{
		constexpr int N = 64, SURF = 32, CS = 16, BALL = 32;
		constexpr float VS = 0.0625f;
		constexpr float BR = (BALL / 2) * VS;	// 内切球半径 1 m

		VoxelObject terrain(N, N, N, VS);
		for (int z = 0; z < N; ++z)
			for (int y = 0; y < SURF; ++y)
				for (int x = 0; x < N; ++x)
					terrain.Cell(x, y, z) = 1;
		terrain.Update();
		static_cast<const VoxelShape *>(terrain.mShape.GetPtr())->GetNumSolidVoxels();

		Array<VoxelObject *> chunk_storage;
		Ref<Shape> compound = sBuildChunkedCompound(terrain, CS, chunk_storage);
		Trace("复合 chunk 地形：%d 个子形状（单网格对照 1 个）", int(chunk_storage.size()));
		// ★ 复合体的 COM = 子形状质量加权平均（子形状 COM 都在自己网格中心），
		//   不在网格中心 ⇒ 查询时把复合体摆在 +C 处，网格中心才回到原点（C# 建车身同理）。
		Vec3 compound_com = compound->GetCenterOfMass();

		VoxelObject ball(BALL, BALL, BALL, VS);
		sMakeSphere(ball, BR);
		ball.Update();

		const float surface_y = SURF * VS - 0.5f * N * VS;	// 地形局部坐标（原点=网格中心）的地表平面
		Quat rot = Quat::sRotation(Vec3(0.3f, 0.7f, 0.2f).Normalized(), 1.1f);	// 落地翻滚的姿态

		// 顺序无关哈希收集器（不含 shape ID：两方案的 SubShapeID 编码本来不同）
		struct Acc : public CollideShapeCollector
		{
			int		mCount = 0;
			float	mDepth = 0, mX = 0, mY = 0, mZ = 0;
			Array<float>			mDepths;
			void	Reset() { mCount = 0; mDepth = mX = mY = mZ = 0; mDepths.clear(); }
			virtual void AddHit(const CollideShapeResult &inR) override
			{
				++mCount;
				mDepth += inR.mPenetrationDepth;
				mX += inR.mContactPointOn1.GetX();
				mY += inR.mContactPointOn1.GetY();
				mZ += inR.mContactPointOn1.GetZ();
				mDepths.push_back(inR.mPenetrationDepth);
			}
		};
		auto query = [&](const Shape *inTerrainShape, Vec3Arg inTerrainTranslation, float inPen, Acc &out)
		{
			out.Reset();
			RVec3 pos(0, surface_y + BR - inPen, 0);
			// 复合体的网格中心在 +COM 处（见上面 compound_com 的说明）；单网格 COM = 0
			CollisionDispatch::sCollideShapeVsShape(ball.mShape, inTerrainShape, Vec3::sOne(), Vec3::sOne(),
				RMat44::sRotationTranslation(rot, pos), RMat44::sTranslation(RVec3(inTerrainTranslation.GetX(), inTerrainTranslation.GetY(), inTerrainTranslation.GetZ())),
				SubShapeIDCreator(), SubShapeIDCreator(), CollideShapeSettings(), out);
		};

		const float pens[] = { 0.5f * VS, 2.0f * VS, 5.0f * VS, -1.0f * VS };
		for (float pen : pens)
		{
			Acc a, b;
			query(terrain.mShape, Vec3::sZero(), pen, a);
			query(compound, compound_com, pen, b);
			Trace("pen %.4f：单网格 %4d 接触 / 复合 %4d 接触；深度和 %.6f vs %.6f",
				pen, a.mCount, b.mCount, double(a.mDepth), double(b.mDepth));
			// 深贯穿时单网格路径在 cMaxVoxelContacts(1024) 截断，而复合路径每 chunk 对各有
			// 1024 预算（深贯穿接触更多）⇒ 只对未截断的姿态要求条数一致
			if (a.mCount < 1024 && b.mCount < 1024)
				CHECK(a.mCount == b.mCount);
			if (a.mCount == b.mCount && a.mCount > 0)
			{
				// 旋转姿态下分块枚举的接触对存在 ±0.04 格级的平移（轴对齐时逐位一致），
				// 深度总差 <5% —— 只做宽松量级校验，明细用 Trace 观察
				CHECK(abs(a.mDepth - b.mDepth) < 0.05f * a.mDepth + 1e-3f);
				Array<float> da = a.mDepths, db = b.mDepths;
				std::sort(da.begin(), da.end(), [](float x, float y) { return x > y; });
				std::sort(db.begin(), db.end(), [](float x, float y) { return x > y; });
				float maxdiff = 0;
				for (int i = 0; i < a.mCount; ++i)
					maxdiff = max(maxdiff, abs(da[i] - db[i]));
				Trace("  最大单条深度差 %.6f（%.2f 格）", double(maxdiff), double(maxdiff / VS));
			}

			// 性能：各 6 轮取中位
			double t1[6], t2[6];
			for (int i = 0; i < 6; ++i)
			{
				Acc w;
				chrono::high_resolution_clock::time_point q0 = chrono::high_resolution_clock::now();
				query(terrain.mShape, Vec3::sZero(), pen, w);
				t1[i] = double(chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - q0).count()) / 1000.0;
				q0 = chrono::high_resolution_clock::now();
				query(compound, compound_com, pen, w);
				t2[i] = double(chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - q0).count()) / 1000.0;
			}
			for (int i = 1; i < 6; ++i)
			{
				double v = t1[i]; int j = i - 1; while (j >= 0 && t1[j] > v) { t1[j + 1] = t1[j]; --j; } t1[j + 1] = v;
				v = t2[i]; j = i - 1; while (j >= 0 && t2[j] > v) { t2[j + 1] = t2[j]; --j; } t2[j + 1] = v;
			}
			Trace("  耗时：单网格 %9.1f us / 复合 %9.1f us（中位）", t1[3], t2[3]);
		}

		// 轴对齐球（无旋转）：两方案必须逐位一致（旋转姿态的 ±0.04 格平移只在旋转时出现）
		{
			Acc a, b;
			RVec3 pos(0, surface_y + BR - 2.0f * VS, 0);
			auto q2 = [&](const Shape *inT, Vec3Arg inTt, Acc &out)
			{
				out.Reset();
				CollisionDispatch::sCollideShapeVsShape(ball.mShape, inT, Vec3::sOne(), Vec3::sOne(),
					RMat44::sTranslation(pos), RMat44::sTranslation(RVec3(inTt.GetX(), inTt.GetY(), inTt.GetZ())),
					SubShapeIDCreator(), SubShapeIDCreator(), CollideShapeSettings(), out);
			};
			q2(terrain.mShape, Vec3::sZero(), a);
			q2(compound, compound_com, b);
			Trace("identity rot pen %.4f：单网格 %d / 复合 %d 接触；深度和 %.6f vs %.6f",
				2.0f * VS, a.mCount, b.mCount, double(a.mDepth), double(b.mDepth));
			CHECK(a.mCount == b.mCount);
			CHECK(a.mDepth == b.mDepth);
		}

		for (VoxelObject *o : chunk_storage)
			delete o;
	}

	/// 旋转网格"薄板列枚举"快路径（2b / 4c-pre）的等价性 + 性能。
	/// 轴对齐地形 x 带旋转的球，四条随机旋转姿态 x 四种穿透深度（悬空/贴地/浅压/深压）各跑两遍：
	///   A  override=0, fast=1  -> 快路径（列枚举，遍历旋转侧 shape1）
	///   B  override=1, fast=1  -> 原路径（强制遍历 shape1 = 同一侧的肥盒遍历）
	/// A 与 B 遍历同一侧、同一套跨格枚举 ⇒ 接触集合必须完全一致；访问顺序不同 ⇒
	/// 用顺序无关哈希（命中量的混合值求和 + 异或）对照。末尾另量一条快路径 vs 原路径的性能对照。
	TEST_CASE("VoxelShapeRotatedSlabExact")
	{
		constexpr int cTerrainN = 128;
		constexpr int cSurface = 100;
		constexpr int cBallN = 80;
		constexpr float cBallR = 40.0f;
		const float half = 0.5f * float(cTerrainN) * cVoxelSize;

		VoxelObject terrain(cTerrainN, cTerrainN, cTerrainN, cVoxelSize);
		for (int z = 0; z < cTerrainN; ++z)
			for (int y = 0; y < cSurface; ++y)
				for (int x = 0; x < cTerrainN; ++x)
					terrain.Cell(x, y, z) = 1;
		terrain.Update();
		((const VoxelShape *)terrain.mShape.GetPtr())->GetNumSolidVoxels();

		VoxelObject ball(cBallN, cBallN, cBallN, cVoxelSize);
		{
			const float h = 0.5f * float(cBallN);
			for (int z = 0; z < cBallN; ++z)
				for (int y = 0; y < cBallN; ++y)
					for (int x = 0; x < cBallN; ++x)
					{
						float cx = (x + 0.5f) - h, cy = (y + 0.5f) - h, cz = (z + 0.5f) - h;
						if (cx * cx + cy * cy + cz * cz <= cBallR * cBallR)
							ball.Cell(x, y, z) = 1;
					}
			ball.Update();
		}

		// 顺序无关哈希收集器：混合值求和（交换律）+ 异或，两条路径的访问顺序不同也必须相等
		struct Acc : public CollideShapeCollector
		{
			uint64			mSum = 0;
			uint64			mXor = 1469598103934665603ull;
			int				mCount = 0;

			static uint64	MixU(uint32 inV) { return (1469598103934665603ull ^ uint64(inV)) * 1099511628211ull; }
			static uint64	MixV(Vec3Arg inV)
			{
				uint32 u[3];
				for (int k = 0; k < 3; ++k)
				{
					float f = k == 0? inV.GetX() : k == 1? inV.GetY() : inV.GetZ();
					memcpy(&u[k], &f, 4);
				}
				return MixU(u[0]) ^ (MixU(u[1]) << 1) ^ (MixU(u[2]) << 2);
			}
			virtual void	AddHit(const CollideShapeResult &inR) override
			{
				++mCount;
				uint32 d;
				memcpy(&d, &inR.mPenetrationDepth, 4);
				uint64 h = MixV(inR.mContactPointOn1) * 3ull ^ MixV(inR.mContactPointOn2) * 5ull
					^ MixU(d) * 7ull ^ MixU(inR.mSubShapeID1.GetValue()) * 11ull ^ MixU(inR.mSubShapeID2.GetValue()) * 13ull;
				mSum += h;
				mXor ^= h * 0x9E3779B97F4A7C15ull;
			}
		};

		NormalModeScope scope(4);
		CollideShapeSettings settings;
		const float cx_w = 64.0f * cVoxelSize - half;
		const float cz_w = cx_w;

		struct Scene { const char *mName; float mPen; };
		Scene scenes[] =
		{
			{ "hover 2", -2.0f },
			{ "touch  ",  0.0f },
			{ "pen 2  ",  2.0f },
			{ "pen 6  ",  6.0f },
		};

		uint32 seed = 24681357u;
		auto rnd = [&seed]() { seed = seed * 1664525u + 1013904223u; return (seed >> 8) & 0xFFFFFFu; };

		for (const Scene &sc : scenes)
			for (int pose = 0; pose < 4; ++pose)
			{
				Vec3 axis = Vec3(float(rnd() % 1000) - 500.0f, float(rnd() % 1000) - 500.0f, float(rnd() % 1000) - 500.0f).Normalized();
				Quat q = Quat::sRotation(axis, 0.35f + float(pose) * 0.4f + float(rnd() % 100) * 0.001f);
				const float cy_w = (float(cSurface) - sc.mPen + cBallR) * cVoxelSize - half;
				Vec3 shift((float(int(rnd() % 40)) - 20.0f) * cVoxelSize, 0.0f, (float(int(rnd() % 40)) - 20.0f) * cVoxelSize);

				Mat44 t_ball = Mat44::sRotationTranslation(q, Vec3(cx_w, cy_w, cz_w) + shift);
				auto query = [&](CollideShapeCollector &ioCollector)
				{
					CollisionDispatch::sCollideShapeVsShape(ball.mShape, terrain.mShape, Vec3::sOne(), Vec3::sOne(),
						t_ball, Mat44::sTranslation(Vec3::sZero()), SubShapeIDCreator(), SubShapeIDCreator(), settings, ioCollector);
				};

				VoxelShape::sIterateSideOverride = 0;
				VoxelShape::sRotatedFastPath = 1;
				Acc a; query(a);
				VoxelShape::sIterateSideOverride = 1;	// 强制遍历 shape1 = 球（旋转侧）：原肥盒遍历的同一侧对照
				Acc b; query(b);
				VoxelShape::sIterateSideOverride = 0;
				VoxelShape::sRotatedFastPath = 1;

				// 截断（cMaxVoxelContacts = 1024）时两条路径保留的子集本来就随访问顺序不同，
				// 只对未截断的姿态做严格对照
				if (a.mCount < 1024 && b.mCount < 1024)
				{
					CHECK(a.mCount == b.mCount);
					CHECK(a.mSum == b.mSum);
					CHECK(a.mXor == b.mXor);
				}
			}

		// 性能对照：贴地姿态（pen 2），快路径 vs 原路径自动选侧
		{
			Quat q = Quat::sRotation(Vec3(0.3f, 1.0f, 0.2f).Normalized(), 1.1f);
			const float cy_w = (float(cSurface) - 2.0f + cBallR) * cVoxelSize - half;
			Mat44 t_ball = Mat44::sRotationTranslation(q, Vec3(cx_w, cy_w, cz_w));
			auto query = [&](CollideShapeCollector &ioCollector)
			{
				CollisionDispatch::sCollideShapeVsShape(ball.mShape, terrain.mShape, Vec3::sOne(), Vec3::sOne(),
					t_ball, Mat44::sTranslation(Vec3::sZero()), SubShapeIDCreator(), SubShapeIDCreator(), settings, ioCollector);
			};
			double samples[2][6];
			int hits = 0;
			for (int v = 0; v < 2; ++v)
			{
				VoxelShape::sRotatedFastPath = v == 0? 1 : 0;
				CountingCollideShapeCollector warm;
				query(warm);
				hits = warm.mCount;
				for (int r = 0; r < 6; ++r)
				{
					chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
					for (int i = 0; i < 20; ++i)
					{
						CountingCollideShapeCollector counting;
						query(counting);
					}
					samples[v][r] = double(chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - start).count()) / (1000.0 * 20);
				}
				for (int i = 1; i < 6; ++i)
				{
					double x = samples[v][i]; int j = i - 1;
					while (j >= 0 && samples[v][j] > x) { samples[v][j + 1] = samples[v][j]; --j; }
					samples[v][j + 1] = x;
				}
			}
			VoxelShape::sRotatedFastPath = 1;
			Trace("rotated pen 2：快路径 %9.1f us / 原路径 %9.1f us（中位，%d hits）",
				samples[0][3], samples[1][3], hits);
		}
	}

	TEST_CASE("VoxelShapeMode4BitExact")
	{
		constexpr int cN = 24;
		constexpr float cVs = 0.05f;
		uint32 seed = 987654321u;
		auto rnd = [&seed]() { seed = seed * 1664525u + 1013904223u; return (seed >> 8) & 0xFFFFFFu; };

		VoxelObject a(cN, cN, cN, cVs);
		VoxelObject b(6, 6, 6, cVs);
		for (int z = 0; z < cN; ++z)
			for (int y = 0; y < cN; ++y)
				for (int x = 0; x < cN; ++x)
					a.Cell(x, y, z) = (y < cN - 2 && (rnd() % 100) < 60)? 1 : 0;
		for (int z = 0; z < 6; ++z)
			for (int y = 0; y < 6; ++y)
				for (int x = 0; x < 6; ++x)
					b.Cell(x, y, z) = 1;
		a.Update();
		b.Update();

		NormalModeScope scope(4);
		CollideShapeSettings settings;

		// 同样的 96 组姿态跑两遍：sBlockMaskCache = 0（每格对重建 2x2x2 块）与 1（复用上一次的块）。
		// 两者**必须逐位相同** —— 缓存只是避免重算，块本身只取决于 (候选体素数据, lo4)。
		auto run = [&](int inCache, uint64 &outHash) -> int
		{
			VoxelShape::sBlockMaskCache = inCache;

			uint64 h = 1469598103934665603ull;
			auto mix = [&h](float inV) { uint32 u; memcpy(&u, &inV, 4); h = (h ^ u) * 1099511628211ull; };
			auto mixu = [&h](uint32 inV) { h = (h ^ inV) * 1099511628211ull; };

			int total = 0;
			for (int i = 0; i < 96; ++i)
			{
				Quat q = Quat::sRotation(Vec3(0.3f, 1.0f, 0.2f).Normalized(), float(i) * 0.37f);
				Vec3 pos((float(int(rnd() % 200)) - 100.0f) * 0.008f,
						 0.25f + float(rnd() % 90) * 0.008f,
						 (float(int(rnd() % 200)) - 100.0f) * 0.008f);
				Mat44 t_b = Mat44::sRotationTranslation(q, pos);

				AllHitCollisionCollector<CollideShapeCollector> collector;
				sCollide(b.mShape, t_b, a.mShape, Mat44::sIdentity(), settings, collector);

				for (const CollideShapeResult &r : collector.mHits)
				{
					mixu(r.mSubShapeID1.GetValue());
					mixu(r.mSubShapeID2.GetValue());
					mix(r.mPenetrationAxis.GetX()); mix(r.mPenetrationAxis.GetY()); mix(r.mPenetrationAxis.GetZ());
					mix(r.mPenetrationDepth);
					mix(r.mContactPointOn1.GetX()); mix(r.mContactPointOn1.GetY()); mix(r.mContactPointOn1.GetZ());
					mix(r.mContactPointOn2.GetX()); mix(r.mContactPointOn2.GetY()); mix(r.mContactPointOn2.GetZ());
					++total;
				}
			}
			outHash = h;
			return total;
		};

		// 注意：rnd 有状态，两遍必须从同一个种子开始
		uint32 saved_seed = seed;
		uint64 hash_nocache = 0, hash_cache = 0;
		int hits_nocache = run(0, hash_nocache);
		seed = saved_seed;
		int hits_cache = run(1, hash_cache);
		VoxelShape::sBlockMaskCache = 1;

		Trace("BITEXACT hits=%d hash=%016llx  |  cache off: hits=%d hash=%016llx",
			hits_cache, (unsigned long long)hash_cache, hits_nocache, (unsigned long long)hash_nocache);
		CHECK(hits_cache == hits_nocache);
		CHECK(hash_cache == hash_nocache);
	}


	TEST_CASE("VoxelShapeCenterOfMassOnlyMovesPivot")
	{
		const float vs = cVoxelSize;
		const Vec3 com(0.0f, -3.5f * vs, 0.0f);			// 实心中心相对盒心（未缩放局部、米）

		VoxelObject piece0(8, 16, 8, vs);				// 质心 = 盒心（旧行为）
		VoxelObject piecec(8, 16, 8, vs);				// 质心 = 实心中心
		VoxelObject *pieces[2] = { &piece0, &piecec };
		for (VoxelObject *o : pieces)
		{
			for (int z = 0; z < 8; ++z)
				for (int y = 0; y < 9; ++y)
					for (int x = 0; x < 8; ++x)
						o->Cell(x, y, z) = 1;
			o->Update();
		}
		VoxelShape *shape0 = const_cast<VoxelShape *>(static_cast<const VoxelShape *>(piece0.mShape.GetPtr()));
		VoxelShape *shapec = const_cast<VoxelShape *>(static_cast<const VoxelShape *>(piecec.mShape.GetPtr()));
		shapec->SetCenterOfMass(com);

		// ---- (1) 声明生效 ----
		CHECK(shape0->GetCenterOfMass() == Vec3::sZero());
		CHECK(shapec->GetCenterOfMass() == com);

		// ---- (2) 几何不动：质心空间的局部包围盒整体 -com（等价于世界位置不变）----
		AABox bounds0 = shape0->GetLocalBounds();
		AABox boundsc = shapec->GetLocalBounds();
		CHECK_APPROX_EQUAL(boundsc.mMin, bounds0.mMin - com, 1.0e-6f);
		CHECK_APPROX_EQUAL(boundsc.mMax, bounds0.mMax - com, 1.0e-6f);

		// ---- (3) 惯量：绕质心 = 绕盒心按平行轴定理反搬 ----
		MassProperties mp0 = shape0->GetMassProperties();
		MassProperties mpc = shapec->GetMassProperties();
		CHECK(mpc.mMass == mp0.mMass);
		Mat44 parallel_axis = mpc.mMass * (Mat44::sScale(com.Dot(com)) - Mat44::sOuterProduct(com, com));
		CHECK_APPROX_EQUAL(mpc.mInertia + parallel_axis, mp0.mInertia, 1.0e-3f);
		// 偏移沿 Y ⇒ 只有绕 X / Z 的惯量变小（平行轴定理：d 方向上的分量不变）
		CHECK(mpc.mInertia.GetDiagonal3().GetX() < mp0.mInertia.GetDiagonal3().GetX());
		CHECK_APPROX_EQUAL(mpc.mInertia.GetDiagonal3().GetY(), mp0.mInertia.GetDiagonal3().GetY(), 1.0e-4f);

		// ---- (4) 支点真的是质心：无重力 / 无初速，只给角速度 ⇒ 质心不动、盒心绕着它转 ----
		{
			PhysicsTestContext c;
			c.GetSystem()->SetGravity(Vec3::sZero());
			BodyInterface &bi = c.GetBodyInterface();
			BodyID id = bi.CreateAndAddBody(BodyCreationSettings(piecec.mShape, RVec3(0, 0, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);
			CHECK_APPROX_EQUAL(Vec3(bi.GetPosition(id)), Vec3::sZero(), 1.0e-5f);			// 渲染锚点仍是盒心
			CHECK_APPROX_EQUAL(Vec3(bi.GetCenterOfMassPosition(id)), com, 1.0e-5f);			// 支点 = 质心

			bi.SetAngularVelocity(id, Vec3(0, 0, 10.0f));
			for (int step = 0; step < 15; ++step)
				c.SimulateSingleStep();

			Vec3 com_pos = Vec3(bi.GetCenterOfMassPosition(id));
			Vec3 box_pos = Vec3(bi.GetPosition(id));
			Trace("质心偏移自转: 质心=(%.4f %.4f %.4f) 盒心=(%.4f %.4f %.4f)", com_pos.GetX(), com_pos.GetY(), com_pos.GetZ(), box_pos.GetX(), box_pos.GetY(), box_pos.GetZ());
			CHECK_APPROX_EQUAL(com_pos, com, 2.0e-3f);						// 无外力 ⇒ 质心原地不动
			CHECK(abs(box_pos.GetX()) > 1.0e-2f);							// 盒心被旋转带着走 ≈ |com.y| * sin(wt) = 0.131
		}

		// ---- (5) 接触不变：同一个世界位置上的探针，两种质心报出来的接触必须一致 ----
		{
			// 形状放在原点时：盒心 y = 0.5，实心占 y 索引 0..8 ⇒ 实心底面在世界 y = -0.5。
			// 用一个半边长 0.05 的探针从下面顶进去 0.03。
			RefConst<Shape> probe = new BoxShape(Vec3(0.05f, 0.05f, 0.05f));
			const float solid_bottom_y = -0.5f;
			Mat44 probe_transform = Mat44::sTranslation(Vec3(0, solid_bottom_y - 0.05f + 0.03f, 0));

			CollideShapeSettings settings;
			AllHitCollisionCollector<CollideShapeCollector> hit0, hitc;
			sCollide(piece0.mShape, Mat44::sIdentity(), probe, probe_transform, settings, hit0);
			// COM = com 的体：Body 的 mPosition = 盒心 + com ⇒ 传进来的质心变换整体平移 +com
			sCollide(piecec.mShape, Mat44::sTranslation(com), probe, probe_transform, settings, hitc);

			Trace("质心偏移接触: COM=0 %d 个，COM=c %d 个", int(hit0.mHits.size()), int(hitc.mHits.size()));
			CHECK(hit0.mHits.size() > 0);
			CHECK(hit0.mHits.size() == hitc.mHits.size());
			if (!hit0.mHits.empty() && hit0.mHits.size() == hitc.mHits.size())
				for (size_t i = 0; i < hit0.mHits.size(); ++i)
				{
					CHECK_APPROX_EQUAL(hit0.mHits[i].mContactPointOn2, hitc.mHits[i].mContactPointOn2, 1.0e-4f);
					CHECK_APPROX_EQUAL(hit0.mHits[i].mPenetrationDepth, hitc.mHits[i].mPenetrationDepth, 1.0e-4f);
				}
		}
	}


	// 8 对齐外扩盒 + 质心偏移的**动力学**回归：1 格实心（盒 8x8x8）落到地面不许被弹飞。
	//
	// 用户实测：质心修好之后，1 格体素碎块 / 1 格柱子的碎块一落地就"弹飞"。
	// 两类碎块的共同点是"实心相对盒子极小"（质心偏移 3.5 格 = 56 倍体素宽），
	// 大碎块偏移不到 1 格所以看不出来。这里同时跑 COM=0 与 COM=实心中心做对照。
	TEST_CASE("VoxelShapeCenterOfMassTinyPieceDrop")
	{
		constexpr float vs = 0.0625f;

		VoxelObject terrain(24, 24, 24, vs);
		sMakeCube(terrain);
		VoxelShape *terrain_shape = const_cast<VoxelShape *>(static_cast<const VoxelShape *>(terrain.mShape.GetPtr()));
		terrain_shape->SetBorderContinues(true);
		const float floor_y = 12.0f * vs;					// 地形顶面 = +half

		// variant 0 = 单格；1 = 1x1x8 柱子（沿 Y）
		for (int variant = 0; variant < 2; ++variant)
			for (int use_com = 0; use_com < 2; ++use_com)
			{
				VoxelObject piece(8, 8, 8, vs);
				if (variant == 0)
					piece.Cell(0, 0, 0) = 1;
				else
					for (int y = 0; y < 8; ++y)
						piece.Cell(0, y, 0) = 1;
				piece.Update();

				VoxelShape *shape = const_cast<VoxelShape *>(static_cast<const VoxelShape *>(piece.mShape.GetPtr()));
				Vec3 com = variant == 0
					? shape->GetVoxelCenter(0, 0, 0)
					: Vec3(shape->GetVoxelCenter(0, 0, 0).GetX(), 0.0f, shape->GetVoxelCenter(0, 0, 0).GetZ());
				shape->SetCenterOfMass(use_com != 0? com : Vec3::sZero());

				MassProperties mp = shape->GetMassProperties();
				Vec3 diag = mp.mInertia.GetDiagonal3();
				Vec3 c0 = mp.mInertia.GetColumn3(0), c1 = mp.mInertia.GetColumn3(1), c2 = mp.mInertia.GetColumn3(2);
				Trace("微碎块 v=%d com=%d 质量 %.6f kg 惯量对角 (%.6e %.6e %.6e)",
					variant, use_com, mp.mMass, diag.GetX(), diag.GetY(), diag.GetZ());
				Trace("   惯量张量 [%.6e %.6e %.6e] [%.6e %.6e %.6e] [%.6e %.6e %.6e]",
					c0.GetX(), c1.GetX(), c2.GetX(), c0.GetY(), c1.GetY(), c2.GetY(), c0.GetZ(), c1.GetZ(), c2.GetZ());

				// ★ 张量必须**正定**（非对角为 0、对角 > 0）。
				//   曾经按完整矩阵搬平行轴，非对角残留 +m*ci*cj（1 格体素 = 1.17e-2，比对角大 73 倍）
				//   ⇒ 主值变成负的 ⇒ Jolt invInertia 为负 ⇒ 落地弹飞。这条断言就是钉它的。
				if (use_com != 0)
				{
					CHECK_APPROX_EQUAL(c0.GetY(), 0.0f, 1.0e-12f);
					CHECK_APPROX_EQUAL(c0.GetZ(), 0.0f, 1.0e-12f);
					CHECK_APPROX_EQUAL(c1.GetX(), 0.0f, 1.0e-12f);
					CHECK_APPROX_EQUAL(c1.GetZ(), 0.0f, 1.0e-12f);
					CHECK_APPROX_EQUAL(c2.GetX(), 0.0f, 1.0e-12f);
					CHECK_APPROX_EQUAL(c2.GetY(), 0.0f, 1.0e-12f);
					// 单格 / 1x1x8 柱子绕 X 的惯量都必须显著大于 0（不是被 clamp 出来的下限）
					CHECK(diag.GetX() > 10.0f * 1.0e-3f * mp.mMass * vs * vs);
				}
				CHECK(diag.GetX() > 0.0f);
				CHECK(diag.GetY() > 0.0f);
				CHECK(diag.GetZ() > 0.0f);

				PhysicsTestContext c;
				c.CreateBody(BodyCreationSettings(terrain.mShape, RVec3(0, 0, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);
				BodyInterface &bi = c.GetBodyInterface();

				const float solid_bottom_local = -4.0f * vs;		// 实心最低点相对盒心
				const float start_center_y = floor_y - solid_bottom_local + 0.20f;
				BodyID id = bi.CreateAndAddBody(BodyCreationSettings(piece.mShape, RVec3(0, start_center_y, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

				float ymax = -1.0e9f, vmax = 0.0f, wmax = 0.0f;
				for (int step = 0; step < 240; ++step)
				{
					c.SimulateSingleStep();
					Vec3 p = Vec3(bi.GetPosition(id));
					ymax = max(ymax, p.GetY());
					vmax = max(vmax, bi.GetLinearVelocity(id).Length());
					wmax = max(wmax, bi.GetAngularVelocity(id).Length());
				}
				Vec3 end = Vec3(bi.GetPosition(id));
				Trace("微碎块 v=%d com=%d 结束 盒心=(%.4f %.4f %.4f) 最高 %.4f 峰值 |v|=%.3f |w|=%.3f",
					variant, use_com, end.GetX(), end.GetY(), end.GetZ(), ymax, vmax, wmax);

				CHECK(vmax < 3.0f);
				CHECK(wmax < 20.0f);
				CHECK(ymax < start_center_y + 0.5f);
			}
	}

	// 碎块 vs 碎块：两个都带质心偏移时窄相位还准不准（真实场景一堆碎块互相撞）。
	TEST_CASE("VoxelShapeCenterOfMassTinyPiecesVsEachOther")
	{
		constexpr float vs = 0.0625f;

		// 底层：8x8x8 盒，只有 (0,0,0) 一格；上层：同样只有一格
		VoxelObject lower(8, 8, 8, vs), upper(8, 8, 8, vs);
		lower.Cell(0, 0, 0) = 1;
		upper.Cell(0, 0, 0) = 1;
		lower.Update();
		upper.Update();
		VoxelShape *ls = const_cast<VoxelShape *>(static_cast<const VoxelShape *>(lower.mShape.GetPtr()));
		VoxelShape *us = const_cast<VoxelShape *>(static_cast<const VoxelShape *>(upper.mShape.GetPtr()));
		Vec3 com = ls->GetVoxelCenter(0, 0, 0);
		ls->SetCenterOfMass(com);
		us->SetCenterOfMass(com);

		PhysicsTestContext c;
		BodyInterface &bi = c.GetBodyInterface();

		// 两个体的盒心都放在 y = 0 附近，实心格底 = 盒心 - 4vs
		BodyID a = bi.CreateAndAddBody(BodyCreationSettings(lower.mShape, RVec3(0, 4.0f * vs, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);
		BodyID b = bi.CreateAndAddBody(BodyCreationSettings(upper.mShape, RVec3(0, 4.0f * vs + vs + 0.10f, 0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);
		(void)a;

		float ymax = -1.0e9f, vmax = 0.0f, wmax = 0.0f;
		for (int step = 0; step < 240; ++step)
		{
			c.SimulateSingleStep();
			Vec3 p = Vec3(bi.GetPosition(b));
			ymax = max(ymax, p.GetY());
			vmax = max(vmax, bi.GetLinearVelocity(b).Length());
			wmax = max(wmax, bi.GetAngularVelocity(b).Length());
			if (step % 60 == 59)
				Trace("叠块 t=%.2f 盒心=(%.4f %.4f %.4f) |v|=%.3f |w|=%.3f", (step + 1) / 60.0f, p.GetX(), p.GetY(), p.GetZ(), bi.GetLinearVelocity(b).Length(), bi.GetAngularVelocity(b).Length());
		}
		Vec3 end = Vec3(bi.GetPosition(b));
		Trace("叠块结束 盒心=(%.4f %.4f %.4f) 最高 %.4f 峰值 |v|=%.3f |w|=%.3f（静止应 y ≈ %.4f）",
			end.GetX(), end.GetY(), end.GetZ(), ymax, vmax, wmax, 4.0f * vs + vs);
		CHECK(vmax < 3.0f);
		CHECK(wmax < 20.0f);
	}

	// 带旋转的落地：负惯量（旧版平行轴搬错）只有在**有力矩**时才会露出来。
	// 单格碎块给 20 rad/s 的自转 + 偏 0.05 m 落下，必须收敛到不动，不许越转越快 / 弹飞。
	TEST_CASE("VoxelShapeCenterOfMassTinyPieceSpinningDrop")
	{
		constexpr float vs = 0.0625f;

		// 台子放大到 96 格（6 m）：揉搓 20 rad/s 的碎块会自己滚出去，24 格（1.5 m）不够走
		VoxelObject terrain(96, 96, 96, vs);
		sMakeCube(terrain);
		VoxelShape *terrain_shape = const_cast<VoxelShape *>(static_cast<const VoxelShape *>(terrain.mShape.GetPtr()));
		terrain_shape->SetBorderContinues(true);

		VoxelObject piece(8, 8, 8, vs);
		piece.Cell(0, 0, 0) = 1;
		piece.Update();
		VoxelShape *shape = const_cast<VoxelShape *>(static_cast<const VoxelShape *>(piece.mShape.GetPtr()));
		shape->SetCenterOfMass(shape->GetVoxelCenter(0, 0, 0));

		PhysicsTestContext c;
		c.CreateBody(BodyCreationSettings(terrain.mShape, RVec3(0, 0, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);
		BodyInterface &bi = c.GetBodyInterface();

		const float floor_y = 48.0f * vs;
		BodyID id = bi.CreateAndAddBody(BodyCreationSettings(piece.mShape, RVec3(0.05, floor_y + 4.0f * vs + 0.25f, 0.0), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);
		bi.SetAngularVelocity(id, Vec3(0, 0, 20.0f));

		float vmax = 0.0f, wmax = 0.0f;
		for (int step = 0; step < 300; ++step)
		{
			c.SimulateSingleStep();
			vmax = max(vmax, bi.GetLinearVelocity(id).Length());
			wmax = max(wmax, bi.GetAngularVelocity(id).Length());
			if (step % 60 == 59)
			{
				Vec3 p = Vec3(bi.GetPosition(id));
				Quat q = bi.GetRotation(id);
				Vec3 up = q * Vec3(0, 1, 0);
				Trace("自转落体 t=%.2f 盒心=(%.4f %.4f %.4f) |v|=%.3f |w|=%.3f 倾角=%.0f°", (step + 1) / 60.0f,
					p.GetX(), p.GetY(), p.GetZ(), bi.GetLinearVelocity(id).Length(), bi.GetAngularVelocity(id).Length(),
					RadiansToDegrees(acos(clamp(up.GetY(), -1.0f, 1.0f))));
			}
		}
		Vec3 end = Vec3(bi.GetPosition(id));
		Vec3 w = bi.GetAngularVelocity(id);
		Vec3 v = bi.GetLinearVelocity(id);
		Trace("自转落体结束 盒心=(%.4f %.4f %.4f) |v|=%.4f |w|=%.4f 峰值 |v|=%.3f |w|=%.3f",
			end.GetX(), end.GetY(), end.GetZ(), v.Length(), w.Length(), vmax, wmax);

		// 落定后必须停住（负惯量时这里会是 |w| 持续增长 / 位置乱飞）
		CHECK(v.Length() < 0.05f);
		CHECK(w.Length() < 0.5f);
		CHECK(vmax < 4.0f);
		CHECK(wmax < 60.0f);
	}

	// 诊断：mode 1 下"边先着地"会不会把方块转起来（用户实测：只要是边先接触地面就旋转）。
	TEST_CASE("VoxelShapeMode1EdgeLanding")
	{
		constexpr float vs = 0.0625f;
		VoxelObject terrain(48, 48, 48, vs);
		sMakeCube(terrain);
		VoxelShape *terrain_shape = const_cast<VoxelShape *>(static_cast<const VoxelShape *>(terrain.mShape.GetPtr()));
		terrain_shape->SetBorderContinues(true);
		const float floor_y = 24.0f * vs;

		// 破坏系统里的典型碎块：0 = 单格 / 1 = 1x1x2 柱 / 2 = L 形 4 格 / 3 = 4x4x4 块
		const int modes[] = { 0, 1, 2, 3, 4 };
		for (int mode_i = 0; mode_i < 5; ++mode_i)
		{
			NormalModeScope scope(modes[mode_i]);
		for (int variant = 0; variant < 4; ++variant)
			for (float tilt : { 0.0f, 5.0f, 30.0f, 45.0f })
				for (float yaw : { 0.0f, 45.0f })
				{
					VoxelObject piece(8, 8, 8, vs);
					switch (variant)
					{
					case 0:
						piece.Cell(3, 0, 3) = 1;
						break;
					case 1:
						piece.Cell(3, 0, 3) = 1;
						piece.Cell(3, 1, 3) = 1;
						break;
					case 2:
						piece.Cell(3, 0, 3) = 1;
						piece.Cell(4, 0, 3) = 1;
						piece.Cell(5, 0, 3) = 1;
						piece.Cell(3, 1, 3) = 1;
						break;
					default:
						for (int z = 0; z < 4; ++z)
							for (int y = 0; y < 4; ++y)
								for (int x = 0; x < 4; ++x)
									piece.Cell(x, y, z) = 1;
						break;
					}
					piece.Update();
					VoxelShape *shape = const_cast<VoxelShape *>(static_cast<const VoxelShape *>(piece.mShape.GetPtr()));

					// 质心 = 实心格中心；同时求实心最低点（相对盒心）
					Vec3 com = Vec3::sZero();
					float lowest = 1.0e9f;
					int n = 0;
					for (int z = 0; z < 8; ++z)
						for (int y = 0; y < 8; ++y)
							for (int x = 0; x < 8; ++x)
								if (piece.Get(x, y, z))
								{
									Vec3 c = shape->GetVoxelCenter(x, y, z);
									com += c;
									lowest = min(lowest, c.GetY() - 0.5f * vs);
									++n;
								}
					com = com / float(n);
					shape->SetCenterOfMass(com);

					PhysicsTestContext c;
					c.CreateBody(BodyCreationSettings(terrain.mShape, RVec3(0, 0, 0), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);
					BodyInterface &bi = c.GetBodyInterface();

					Quat q = Quat::sRotation(Vec3::sAxisZ(), DegreesToRadians(tilt)) * Quat::sRotation(Vec3::sAxisY(), DegreesToRadians(yaw));
					RVec3 start(0, floor_y - lowest + 0.10f, 0);
					BodyID id = bi.CreateAndAddBody(BodyCreationSettings(piece.mShape, start, q, EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

					float wmax = 0.0f;
					for (int step = 0; step < 240; ++step)
					{
						c.SimulateSingleStep();
						wmax = max(wmax, bi.GetAngularVelocity(id).Length());
						if (step % 60 == 59)
						{
							Vec3 up = bi.GetRotation(id) * Vec3(0, 1, 0);
							Trace("m1 v=%d tilt=%.0f yaw=%.0f t=%.1fs angle=%.1f y=%.4f |w|=%.3f", variant, tilt, yaw, (step + 1) / 60.0f,
								RadiansToDegrees(acos(clamp(up.GetY(), -1.0f, 1.0f))), Vec3(bi.GetPosition(id)).GetY(), bi.GetAngularVelocity(id).Length());
						}
					}
					Vec3 up = bi.GetRotation(id) * Vec3(0, 1, 0);
					Vec3 end = Vec3(bi.GetPosition(id));
					Vec3 fwd = bi.GetRotation(id) * Vec3(0, 0, 1);
					Trace("M%d END v=%d tilt=%.0f yaw=%.0f angle=%.1f spin=%.1f y=%.4f dx=%.3f |w|=%.4f wmax=%.2f", modes[mode_i], variant, tilt, yaw,
						RadiansToDegrees(acos(clamp(up.GetY(), -1.0f, 1.0f))), RadiansToDegrees(atan2(fwd.GetX(), fwd.GetZ())) - yaw, end.GetY(), end.GetX(), bi.GetAngularVelocity(id).Length(), wmax);
				}
		}
	}

	TEST_CASE("VoxelShapeMode4HoleJitter")
	{
		// "碎块掉进洞里左右抖"：碎块以某个角度卡进竖井后，井壁格与碎块格既可能**同层横向相邻**
		// （along 投影 ≈ 0），也可能斜对角相邻。这两类格对用井壁的整面法线算 depth 会得到
		// 接近一整格的假穿透 ⇒ 每帧被横推、来回反弹。井宽 = 8 / 10 格，碎块 8^3，倾角 0/10/20°。
		constexpr float vs = 0.0625f;
		constexpr int   cSize = 48;
		const float half = 0.5f * float(cSize) * vs;

		for (int gap = 0; gap <= 1; ++gap)
		{
			const int hole = 8 + 2 * gap;
			const int ox = 20 - gap, oz = 20 - gap;

			VoxelObject terrain(cSize, cSize, cSize, vs);
			sMakeCube(terrain);
			for (int z = oz; z < oz + hole; ++z)
				for (int x = ox; x < ox + hole; ++x)
					for (int y = 16; y < cSize; ++y)
						terrain.Cell(x, y, z) = 0;
			terrain.Update();
			VoxelShape *terrain_shape = const_cast<VoxelShape *>(static_cast<const VoxelShape *>(terrain.mShape.GetPtr()));
			terrain_shape->SetBorderContinues(true);

			VoxelObject piece(8, 8, 8, vs);
			sMakeCube(piece);

			const float cx = (float(ox) + 0.5f * float(hole)) * vs - half;
			const float cz = (float(oz) + 0.5f * float(hole)) * vs - half;
			const float rest_y = 16.0f * vs - half + 0.5f * 8.0f * vs;

			for (float tilt : { 0.0f, 10.0f, 20.0f })
				for (int mode : { 1, 4 })
				{
					NormalModeScope scope(mode);
					Quat q = Quat::sRotation(Vec3::sAxisX(), DegreesToRadians(tilt));

					// 1) 静置姿态的接触集合
					{
						CollideShapeSettings settings;
						AllHitCollisionCollector<CollideShapeCollector> collector;
						sCollide(piece.mShape, Mat44::sTranslation(Vec3(cx, rest_y - 0.005f, cz)) * Mat44::sRotation(q),
							terrain.mShape, Mat44::sTranslation(Vec3::sZero()), settings, collector);

						int horizontal = 0;
						float max_depth = 0.0f;
						for (const CollideShapeResult &h : collector.mHits)
							if (abs(h.mPenetrationAxis.GetY()) < 0.9f)
							{
								++horizontal;
								max_depth = max(max_depth, h.mPenetrationDepth);
							}
						Trace("HOLE gap=%d tilt=%.0f mode=%d contacts=%d horizontal=%d maxHorDepth=%.2f cells",
							gap, tilt, mode, int(collector.mHits.size()), horizontal, max_depth / vs);
					}

					// 2) 3 s 仿真：落定后（t > 1.5 s）的水平峰峰值与 x 速度反向次数
					{
						PhysicsTestContext c;
						c.CreateBody(BodyCreationSettings(terrain.mShape, RVec3::sZero(), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);
						BodyInterface &bi = c.GetBodyInterface();
						BodyID id = bi.CreateAndAddBody(BodyCreationSettings(piece.mShape, RVec3(cx, rest_y + 0.3f, cz), q, EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

						float xmin = 1.0e9f, xmax = -1.0e9f, zmin = 1.0e9f, zmax = -1.0e9f;
						int flips = 0;
						float prev_vx = 0.0f;
						for (int step = 0; step < 360; ++step)
						{
							c.SimulateSingleStep();
							Vec3 p = Vec3(bi.GetPosition(id));
							Vec3 v = bi.GetLinearVelocity(id);
							if (step > 90)
							{
								xmin = min(xmin, p.GetX()); xmax = max(xmax, p.GetX());
								zmin = min(zmin, p.GetZ()); zmax = max(zmax, p.GetZ());
							}
							if (v.GetX() * prev_vx < 0.0f && abs(v.GetX()) > 0.02f)
								++flips;
							prev_vx = v.GetX();
						}
						Vec3 p = Vec3(bi.GetPosition(id));
						Vec3 v = bi.GetLinearVelocity(id);
						Trace("HOLE gap=%d tilt=%.0f mode=%d drift=(%.4f,%.4f) p2p=(%.4f,%.4f) flips=%d v=%.3f dy=%.4f",
							gap, tilt, mode, p.GetX() - cx, p.GetZ() - cz, xmax - xmin, zmax - zmin, flips,
							sqrt(v.GetX() * v.GetX() + v.GetZ() * v.GetZ()), p.GetY() - rest_y);

						// 回归判据：mode 4 在洞里不许"被顶飞 / 穿透"。gap=0（碎块正好塞满洞）允许
						// 半格以内的下沉与微动（体素接触的固有 slop）；gap>=1 必须完全静止。
						if (mode == 4)
						{
							CAPTURE(gap);
							CAPTURE(tilt);
							CHECK(p.GetY() - rest_y < 0.03f);
							CHECK(p.GetY() - rest_y > -0.10f);
							CHECK(sqrt((p.GetX() - cx) * (p.GetX() - cx) + (p.GetZ() - cz) * (p.GetZ() - cz)) < 0.10f);
							CHECK(xmax - xmin < 0.09f);
							CHECK(zmax - zmin < 0.09f);
							if (gap >= 1)
							{
								CHECK(xmax - xmin < 0.005f);
								CHECK(zmax - zmin < 0.005f);
							}
						}
					}
				}

			terrain_shape->SetBorderContinues(false);
		}
	}
}
