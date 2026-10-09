// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include "UnitTestFramework.h"
#include "PhysicsTestContext.h"
#include "Layers.h"
#include <Jolt/Physics/Collision/Shape/VoxelShape.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
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
JPH_SUPPRESS_WARNINGS_STD_END

TEST_SUITE("VoxelShapeTests")
{
	// All shapes in these tests are 10 x 10 x 10 voxels of 0.1 m, so they have a bounding box of 1 x 1 x 1 meters
	constexpr int cNumVoxels = 10;
	constexpr float cVoxelSize = 0.1f;
	constexpr float cHalfExtent = 0.5f * cNumVoxels * cVoxelSize;

	/// Holds the voxel data and the shape that reads it (the shape does not own the data)
	struct VoxelObject
	{
							VoxelObject(int inSizeX, int inSizeY, int inSizeZ, float inVoxelSize) :
								mSizeX(inSizeX), mSizeY(inSizeY), mSizeZ(inSizeZ), mVoxelSize(inVoxelSize)
		{
			mVoxels.resize(inSizeX * inSizeY * inSizeZ, 0);
			mShape = new VoxelShape(inSizeX, inSizeY, inSizeZ, inVoxelSize, mVoxels.data());
		}

		uint8 &				Cell(int inX, int inY, int inZ)
		{
			JPH_ASSERT(inX >= 0 && inX < mSizeX && inY >= 0 && inY < mSizeY && inZ >= 0 && inZ < mSizeZ);
			return mVoxels[inX + inY * mSizeX + inZ * mSizeX * mSizeY];
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
			for (const uint8 &v : mVoxels)
				if (v != 0)
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
		Array<uint8>		mVoxels;
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

	/// Area of a (convex, planar) polygon
	static float sGetPolygonArea(const CollideShapeResult::Face &inFace)
	{
		Vec3 sum = Vec3::sZero();
		for (CollideShapeResult::Face::size_type i = 1; i + 1 < inFace.size(); ++i)
			sum += (inFace[i] - inFace[0]).Cross(inFace[i + 1] - inFace[0]);
		return 0.5f * sum.Length();
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

	TEST_CASE("VoxelShapeVsMeshShape")
	{
		RefConst<Shape> mesh = sCreateMeshFloor();
		CHECK(mesh != nullptr);
		if (mesh == nullptr)
			return;
		CHECK(mesh->GetSubType() == EShapeSubType::Mesh);

		VoxelObject cube(cNumVoxels, cNumVoxels, cNumVoxels, cVoxelSize);
		sMakeCube(cube);

		// The cube is pushed 0.05 into the floor
		{
			CollideShapeSettings settings;
			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(cube.mShape, Mat44::sTranslation(Vec3(0, cHalfExtent - 0.05f, 0)), mesh, Mat44::sIdentity(), settings, collector);
			CHECK(collector.mHits.size() > 0);
			if (!collector.mHits.empty())
			{
				CHECK_APPROX_EQUAL(collector.mHits[0].mPenetrationDepth, 0.05f, 0.002f);
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
			settings.mCollectFacesMode = ECollectFacesMode::CollectFaces;
			AllHitCollisionCollector<CollideShapeCollector> collector;
			sCollide(cube1.mShape, Mat44::sTranslation(Vec3(0, cHalfExtent, 0)), cube2.mShape, Mat44::sTranslation(Vec3(0, 2.0f * cHalfExtent + cHalfExtent - 0.05f, 0)), settings, collector);
			CHECK(collector.mHits.size() > 0);
			if (!collector.mHits.empty())
			{
				const CollideShapeResult &hit = collector.mHits[0];
				CHECK_APPROX_EQUAL(hit.mPenetrationDepth, 0.05f, 0.002f);
				CHECK(abs(hit.mPenetrationAxis.Normalized().GetY()) > 0.99f);

				// A voxel shape generates one contact per cell of the overlap, so the supporting face of a
				// single contact is the face of one cell (0.1 x 0.1 m) and not the whole 1 x 1 m face of the grid
				CHECK(hit.mShape1Face.size() == 4);
				CHECK(hit.mShape2Face.size() == 4);
				CHECK_APPROX_EQUAL(sGetPolygonArea(hit.mShape1Face), cVoxelSize * cVoxelSize, 1.0e-4f);
				CHECK_APPROX_EQUAL(sGetPolygonArea(hit.mShape2Face), cVoxelSize * cVoxelSize, 1.0e-4f);

				// Shape 1 is at y = 0.5 (top face at 1.0), shape 2 at y = 1.45 (bottom face at 0.95)
				for (const Vec3 &v : hit.mShape1Face)
					CHECK_APPROX_EQUAL(v.GetY(), 1.0f, 0.001f);
				for (const Vec3 &v : hit.mShape2Face)
					CHECK_APPROX_EQUAL(v.GetY(), 0.95f, 0.001f);

				// The contacts of the individual cells tile the entire face: 10 x 10 cells of 0.1 x 0.1 m is
				// the 1 m^2 of contact area that a single contact of the convex hull used to report. This is
				// what makes the per cell approach work in the solver: the aggregate of the contacts of the
				// cells is the same as the contact that a solid box would generate.
				// Note that only the contacts of the cells that really overlap are counted. Because both grids
				// have the same cell size and orientation in this test, the cells that are exactly next to each
				// other touch and generate a contact with a depth of 0 as well, which would double count.
				float area1 = 0.0f, area2 = 0.0f;
				int num_overlapping = 0;
				for (const CollideShapeResult &h : collector.mHits)
					if (h.mPenetrationDepth > 0.01f)				// Well above 0 (touching) and below the 0.05 m overlap
					{
						area1 += sGetPolygonArea(h.mShape1Face);
						area2 += sGetPolygonArea(h.mShape2Face);
						++num_overlapping;
					}
				CHECK(num_overlapping == cNumVoxels * cNumVoxels);
				CHECK_APPROX_EQUAL(area1, 1.0f, 0.01f);
				CHECK_APPROX_EQUAL(area2, 1.0f, 0.01f);
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
			sCheckApproxEqual(bi.GetPosition(top).GetY(), 3.0 * cHalfExtent, 0.02);
			CHECK(abs(bi.GetLinearVelocity(top).GetY()) < 0.01f);

			// It should not have rotated (with a single contact point it would tip over)
			Quat rotation = bi.GetRotation(top);
			CHECK(abs(rotation.GetX()) < 0.01f);
			CHECK(abs(rotation.GetY()) < 0.01f);
			CHECK(abs(rotation.GetZ()) < 0.01f);
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
				CHECK_APPROX_EQUAL(collector.mHits[0].mPenetrationDepth, 0.05f, 0.002f);
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
				// diagonal. The deepest contact is the cell that the sphere hits head on and that one has the
				// depth and the axis that the convex hull of the cube used to report for the whole face.
				float max_depth = -FLT_MAX;
				Vec3 deepest_axis = Vec3::sZero();
				for (const CollideShapeResult &h : collector.mHits)
					if (h.mPenetrationDepth > max_depth)
					{
						max_depth = h.mPenetrationDepth;
						deepest_axis = h.mPenetrationAxis.Normalized();
					}
				CHECK_APPROX_EQUAL(max_depth, 0.05f, 0.002f);
				CHECK(abs(deepest_axis.GetX()) > 0.99f);

				// No contact may be deeper than the actual overlap of the two shapes: the sphere is 0.05 m
				// inside the cube, so no individual cell may report more than that
				for (const CollideShapeResult &h : collector.mHits)
					CHECK(h.mPenetrationDepth <= 0.05f + 0.002f);
			}
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
		CHECK_APPROX_EQUAL(shape->GetMassProperties().mMass, original_mass * 0.9f, 1.0e-2f);
		CHECK(voxel_shape->GetNumSolidVoxels() == 900);

		// The bottom of the shape is now one voxel higher so the body falls by one voxel size
		c.Simulate(1.0f);
		sCheckApproxEqual(bi.GetPosition(id).GetY(), cHalfExtent - cVoxelSize, 0.02);
	}

	/// Reports the cost of the narrow phase for voxel shapes and compares it against the equivalent box shapes.
	/// This is not a correctness test (it has no assertions), it exists to be able to judge the cost of the
	/// collision queries.
	///
	/// A voxel shape has no derived geometry (no hull, no mesh): the narrow phase walks the solid voxels in the
	/// overlap region and hands each of them to the regular convex vs convex path as a box (see
	/// VoxelShape::sCollideConvexVsVoxel). So the cost of a voxel vs voxel query is (number of solid voxels in
	/// the overlap) x (cost of one box vs box query), which is exactly what this table shows: compare a 10^3
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
		constexpr int cHoleMin = 6;
		constexpr int cHoleMax = 14;

		// The same wall with a hole as in VoxelShapeConcave: 2 x 0.4 x 2 m, centered on the origin, with a
		// 0.8 x 0.4 x 0.8 m hole through it
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

		// A 0.3 m box, it fits through the hole (the hole is 0.8 m wide and 0.4 m tall)
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
		// 0.4 m in y), so the cast has to run along y to pass through it. The box starts 1 m above the wall
		// and travels 2 m, ending 1 m below it.
		{
			ClosestHitCollisionCollector<CastShapeCollector> collector = cast(Vec3(0, 1.0f, 0), Vec3(0, -2.0f, 0));
			CHECK(!collector.HadHit());
		}

		// Next to the hole (above the solid part of the wall) the box hits the top face of the wall. The wall's
		// top face is at y = 0.2 m and the box is 0.15 m, so the box travels 1.0 - 0.2 - 0.15 = 0.65 m out of
		// the 2 m: a fraction of 0.325.
		{
			ClosestHitCollisionCollector<CastShapeCollector> collector = cast(Vec3(-0.7f, 1.0f, 0), Vec3(0, -2.0f, 0));
			CHECK(collector.HadHit());
			if (collector.HadHit())
			{
				CHECK_APPROX_EQUAL(collector.mHit.mFraction, 0.325f, 0.005f);
				CHECK(abs(collector.mHit.mPenetrationAxis.Normalized().GetY()) > 0.99f);
			}
		}

		// Next to the hole in the other direction too: the box hits the front face of the wall. The face is at
		// z = -1 m and the box is 0.15 m, so the box travels 1.5 - 1.0 - 0.15 = 0.35 m out of the 2 m: 0.175.
		{
			ClosestHitCollisionCollector<CastShapeCollector> collector = cast(Vec3(-0.7f, 0, -1.5f), Vec3(0, 0, 2.0f));
			CHECK(collector.HadHit());
			if (collector.HadHit())
			{
				CHECK_APPROX_EQUAL(collector.mHit.mFraction, 0.175f, 0.005f);
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

		// The wall is only 0.4 m tall, a box that is cast above it does not hit anything
		{
			ClosestHitCollisionCollector<CastShapeCollector> collector = cast(Vec3(-0.7f, 0.5f, -1.5f), Vec3(0, 0, 2.0f));
			CHECK(!collector.HadHit());
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
		constexpr int cHoleMin = 6;
		constexpr int cHoleMax = 14;

		VoxelObject wall(cSize, cSize, cSize, cVoxelSize);

		// A wall (4 voxels thick) with a square hole (8 x 4 x 8 voxels, so 0.8 x 0.4 x 0.8 m) through it
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
		CHECK(num_hits(Vec3(-0.75f, 0.0f, 0.0f)) > 0);

		// The hole goes through the whole wall, so a 0.3 m box travels through it in the other direction as
		// well. The hole is 0.8 m wide, the box is 0.3 m, so the box can move 0.25 m to either side without
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
			CHECK(num_hits(Vec3(-0.7f, 0.0f, z)) > 0);
		}

		// A box that straddles the edge of the hole collides with the solid part of the wall next to it
		CHECK(num_hits(Vec3(-0.35f, 0.0f, 0.0f)) > 0);
		CHECK(num_hits(Vec3(0.0f, 0.0f, -0.35f)) > 0);

		// And outside of the wall slab (which is only 2 x 2 x 0.4 m in x, z, y) there is nothing to hit
		CHECK(num_hits(Vec3(0.0f, 0.0f, 1.3f)) == 0);
		CHECK(num_hits(Vec3(-0.7f, 0.0f, 1.3f)) == 0);
		CHECK(num_hits(Vec3(-0.7f, 0.5f, 0.0f)) == 0);
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
}
