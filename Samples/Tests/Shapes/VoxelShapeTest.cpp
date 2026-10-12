// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <TestFramework.h>

#include <Tests/Shapes/VoxelShapeTest.h>
#include <Jolt/Core/StringTools.h>
#include <Jolt/Geometry/Triangle.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Application/DebugUI.h>
#include <Utils/DebugRendererSP.h>
#include <Layers.h>

JPH_IMPLEMENT_RTTI_VIRTUAL(VoxelShapeTest)
{
	JPH_ADD_BASE_CLASS(VoxelShapeTest, Test)
}

/// Names shown in the settings menu, must match VoxelShapeTest::sVoxelSizes
static const char *sVoxelSizeNames[] = {
	"0.03125 m (0.5 m bounding box)",
	"0.0625 m (1.0 m bounding box)",
	"0.125 m (2.0 m bounding box)"
};

VoxelShapeTest::~VoxelShapeTest()
{
	// The shapes reference the voxel data that we own, so remove the bodies before throwing the voxel data away.
	// Note that SamplesApp deletes the test before it deletes the physics system, so mBodyInterface is still valid here.
	if (mBodyInterface != nullptr)
		for (VoxelObject *object : mObjects)
			if (!object->mBodyID.IsInvalid())
			{
				mBodyInterface->RemoveBody(object->mBodyID);
				mBodyInterface->DestroyBody(object->mBodyID);
			}

	for (VoxelObject *object : mObjects)
		delete object;
}

void VoxelShapeTest::Initialize()
{
	// Size of the bounding box of a voxel object (1 m for the default voxel size of 0.1 m)
	const float u = sGetObjectSize();
	const float half = 0.5f * u;

	// Floor: a box shape, which also covers voxel vs convex shape collisions
	CreateFloor(20.0f * u);

	// Row 1: a voxel cube, sphere and cylinder dropping on the floor
	CreateVoxelObject(EShapeType::Cube, "Cube", RVec3(-6.0f * u, 2.0f * u, 0), EMotionType::Dynamic);
	CreateVoxelObject(EShapeType::Sphere, "Sphere", RVec3(-3.0f * u, 2.0f * u, 0), EMotionType::Dynamic);
	CreateVoxelObject(EShapeType::Cylinder, "Cylinder", RVec3(0.0f * u, 2.0f * u, 0), EMotionType::Dynamic);

	// Voxel vs voxel: a dynamic voxel cube dropping on a static voxel platform
	mPlatform = CreateVoxelObject(EShapeType::Cube, "Platform", RVec3(3.0f * u, half, 0), EMotionType::Static);
	CreateVoxelObject(EShapeType::Cube, "On Platform", RVec3(3.0f * u, 3.5f * u, 0), EMotionType::Dynamic);

	// Voxel vs voxel: a stack of dynamic voxel cubes on a static voxel base
	CreateVoxelObject(EShapeType::Cube, "Stack Base", RVec3(6.0f * u, half, 0), EMotionType::Static);
	CreateVoxelObject(EShapeType::Cube, "Stack 1", RVec3(6.0f * u, 1.5f * u, 0), EMotionType::Dynamic);
	CreateVoxelObject(EShapeType::Cube, "Stack 2", RVec3(6.0f * u, 2.5f * u, 0), EMotionType::Dynamic);
	CreateVoxelObject(EShapeType::Cube, "Stack 3", RVec3(6.0f * u, 3.5f * u, 0), EMotionType::Dynamic);

	// Row 2: voxel vs mesh shape
	{
		// Create a flat grid of triangles
		const int n = 4;
		const float cell = u;
		TriangleList triangles;
		for (int x = 0; x < n; ++x)
			for (int z = 0; z < n; ++z)
			{
				float x1 = (x - 0.5f * n) * cell;
				float z1 = (z - 0.5f * n) * cell;
				float x2 = x1 + cell;
				float z2 = z1 + cell;
				triangles.push_back(Triangle(Float3(x1, 0, z1), Float3(x1, 0, z2), Float3(x2, 0, z1)));
				triangles.push_back(Triangle(Float3(x1, 0, z2), Float3(x2, 0, z2), Float3(x2, 0, z1)));
			}

		// Positioned slightly above the box floor so that the two don't overlap
		mBodyInterface->CreateAndAddBody(BodyCreationSettings(new MeshShapeSettings(triangles), RVec3(-2.0f * u, 0.02f * u, 6.0f * u), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

		CreateVoxelObject(EShapeType::Cube, "On Mesh", RVec3(-2.0f * u, 2.0f * u, 6.0f * u), EMotionType::Dynamic);
	}

	// Row 2: voxel vs height field
	{
		// Create a height field with a bulge in the middle
		const uint n = 5;
		Array<float> heights;
		heights.resize(n * n);
		for (uint y = 0; y < n; ++y)
			for (uint x = 0; x < n; ++x)
			{
				float dx = float(x) - 0.5f * (n - 1);
				float dz = float(y) - 0.5f * (n - 1);
				heights[y * n + x] = 0.1f * (2.0f - (dx * dx + dz * dz));
			}

		// The height field is centered on the origin of the body.
		// Note: the settings object is created on the heap because BodyCreationSettings takes ownership of it.
		HeightFieldShapeSettings *settings = new HeightFieldShapeSettings(heights.data(), Vec3(-0.5f * (n - 1) * u, 0, -0.5f * (n - 1) * u), Vec3::sReplicate(u), n);
		mBodyInterface->CreateAndAddBody(BodyCreationSettings(settings, RVec3(2.0f * u, 0, 6.0f * u), Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

		CreateVoxelObject(EShapeType::Sphere, "On Height Field", RVec3(2.0f * u, 2.0f * u, 6.0f * u), EMotionType::Dynamic);
	}

	// Row 3: concave voxel data. A block with a rectangular shaft straight through it (a wall with a window).
	// The ball that is dropped above the shaft falls all the way through to the floor, the one that is dropped
	// above the solid part of the block lands on top of it. With a convex hull of the voxels (the previous
	// implementation) the hull would fill the shaft and both balls would land on top of the block.
	{
		mWall = CreateVoxelObject(EShapeType::Window, "Wall With Window", RVec3(0, half, 9.0f * u), EMotionType::Static);

		// A small ball (1/10 of the object size, the shaft is 6/16 of it) so that it fits through the shaft
		float ball_radius = 0.1f * u;
		RefConst<Shape> ball = new SphereShape(ball_radius);

		// Above the shaft -> falls through. The shaft is centered on the grid so the ball goes to the center.
		mBodyInterface->CreateAndAddBody(BodyCreationSettings(ball, RVec3(0, 3.0f * u, 9.0f * u), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

		// Above the solid part -> lands on top of the block. The solid part next to the shaft starts at 3/16
		// from the center, so the center of the ball is placed at 3.5/16 from the center (0.35 m, which is
		// still the value it was with the 0.1 m voxels, as the bounding box did not change).
		mBodyInterface->CreateAndAddBody(BodyCreationSettings(ball, RVec3(-0.35f * u, 3.0f * u, 9.0f * u), Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);
	}

	// In front of row 1 (closest to the camera): a voxel donut resting on a voxel cone.
	//
	// This is the stress case for the voxel vs voxel narrow phase: a concave body (the donut has a hole)
	// on a slanted, stepped surface, so cell corners meet cell faces all over the contact. Both grids are
	// also neither cubic nor a multiple of 8 cells (the chunk occupancy mask pads the last chunk).
	{
		const float cell = sGetVoxelSize();

		// The cone is the stand: 24 cells diameter at the base, 24 cells tall (1.5 m, the same size as with
		// the old 0.1 m voxels), with its base on the floor
		constexpr int cConeCells = 24;							// diameter and height
		const float cone_radius = 0.5f * cConeCells;			// in cells
		CreateVoxelObject(EShapeType::Cone, "Cone", RVec3(0, 0.5f * cConeCells * cell, -6.0f * u), EMotionType::Static, cConeCells, cConeCells, cConeCells);

		// The donut comes to rest where the tube (radius cDonutTube) touches the slant of the cone. The slant
		// goes through (radial = cone_radius, height = 0) and (radial = 0, height = cConeCells), so the
		// distance from the tube center line (radial = cDonutRing, at height y) to the slant is:
		//     |cConeCells * cDonutRing + cone_radius * y - cone_radius * cConeCells| / slant_length
		// Setting that to cDonutTube and solving for y gives the resting height. It is dropped from a little
		// above it so that it settles instead of starting in penetration.
		constexpr float cDonutRing = 11.0f;						// radius of the tube center line, in cells
		constexpr float cDonutTube = 5.0f;						// radius of the tube, in cells
		float slant_length = sqrt(Square(cConeCells) + Square(cone_radius));
		float rest_height = (cone_radius * cConeCells + cDonutTube * slant_length - cConeCells * cDonutRing) / cone_radius;
		CreateVoxelObject(EShapeType::Donut, "Donut", RVec3(0, (rest_height + 2.0f) * cell, -6.0f * u), EMotionType::Dynamic, 32, 11, 32);
	}
}

VoxelShapeTest::VoxelObject *VoxelShapeTest::CreateVoxelObject(EShapeType inType, const char *inName, const RVec3 &inPosition, EMotionType inMotionType, int inSizeX, int inSizeY, int inSizeZ)
{
	VoxelObject *object = new VoxelObject;
	object->mName = inName;
	object->mSizeX = inSizeX;
	object->mSizeY = inSizeY;
	object->mSizeZ = inSizeZ;
	object->mTexelsX = (inSizeX + 3) >> 2;
	object->mTexelsY = (inSizeY + 3) >> 2;
	object->mTexelPlane = object->mTexelsX * object->mTexelsY;
	object->mOccupancy.resize(size_t(object->mTexelPlane) * ((inSizeZ + 3) >> 2));
	object->ClearVoxels();

	// Create the shape. It does not copy the voxel data, it reads the occupancy bitmap directly, so
	// that bitmap must stay alive.
	RefConst<Shape> shape = new VoxelShape(uint(inSizeX), uint(inSizeY), uint(inSizeZ), sGetVoxelSize(), object->mOccupancy.data(), cDensity);
	object->mShape = shape;

	// Fill in the voxels
	FillVoxels(*object, inType);

	// Refresh the cached properties (extent of the solid voxels, volume, mass and inertia) after the voxel data was modified
	object->GetVoxelShape()->Recalculate();

	// Create the body
	bool is_static = inMotionType == EMotionType::Static;
	object->mBodyID = mBodyInterface->CreateAndAddBody(
		BodyCreationSettings(object->mShape, inPosition, Quat::sIdentity(), inMotionType, is_static? Layers::NON_MOVING : Layers::MOVING),
		is_static? EActivation::DontActivate : EActivation::Activate);

	mObjects.push_back(object);

	return object;
}

void VoxelShapeTest::FillVoxels(VoxelObject &ioObject, EShapeType inType)
{
	const VoxelShape *shape = ioObject.GetVoxelShape();
	const int size_x = ioObject.mSizeX, size_y = ioObject.mSizeY, size_z = ioObject.mSizeZ;

	if (inType == EShapeType::Cube)
	{
		// Fill the entire grid
		for (int z = 0; z < size_z; ++z)
			for (int y = 0; y < size_y; ++y)
				for (int x = 0; x < size_x; ++x)
					ioObject.SetVoxel(x, y, z, true);
	}
	else if (inType == EShapeType::Window)
	{
		// A block with a rectangular shaft through it. The shaft is 6 x 6 voxels (0.375 m, the same size as
		// with the old 0.1 m voxels, where it was 4 x 4), centered on the grid, so a body that is small
		// enough falls straight through it. Note that this data is concave: there is no way to describe it
		// with a convex hull, which is exactly why the shape is not a convex shape.
		int hole_min = cNumVoxels / 2 - cWindowHalfSize;
		int hole_max = cNumVoxels / 2 + cWindowHalfSize;
		for (int z = 0; z < size_z; ++z)
			for (int y = 0; y < size_y; ++y)
				for (int x = 0; x < size_x; ++x)
					if (x < hole_min || x >= hole_max || z < hole_min || z >= hole_max)
						ioObject.SetVoxel(x, y, z, true);
	}
	else if (inType == EShapeType::Donut)
	{
		// A torus around the local Y axis: the tube center line is a circle of radius cRing in the XZ plane,
		// the tube has radius cTube. Everything is in cells, so the outer diameter is
		// 2 * (cRing + cTube) = 2 * (11 + 5) = 32 cells (2.0 m, the same size as with the old 0.1 m voxels)
		// and the tube is 2 * cTube + 1 = 11 cells thick (hence the 32 x 11 x 32 grid).
		constexpr float cRing = 11.0f;
		constexpr float cTube = 5.0f;
		for (int z = 0; z < size_z; ++z)
			for (int y = 0; y < size_y; ++y)
				for (int x = 0; x < size_x; ++x)
				{
					// Cell center relative to the center of the grid, in cells
					float cx = float(x) + 0.5f - 0.5f * float(size_x);
					float cy = float(y) + 0.5f - 0.5f * float(size_y);
					float cz = float(z) + 0.5f - 0.5f * float(size_z);

					float dr = sqrt(Square(cx) + Square(cz)) - cRing;
					if (Square(dr) + Square(cy) <= Square(cTube))
						ioObject.SetVoxel(x, y, z, true);
				}
	}
	else if (inType == EShapeType::Cone)
	{
		// A cone with the apex up (along the local Y axis): the radius shrinks linearly from cBaseDiameter / 2
		// at the bottom to 0 at the top.
		constexpr float cBaseDiameter = 24.0f;
		for (int z = 0; z < size_z; ++z)
			for (int y = 0; y < size_y; ++y)
				for (int x = 0; x < size_x; ++x)
				{
					// 0 at the base, 1 at the tip
					float h = (float(y) + 0.5f) / float(size_y);
					float radius = 0.5f * cBaseDiameter * (1.0f - h);

					float cx = float(x) + 0.5f - 0.5f * float(size_x);
					float cz = float(z) + 0.5f - 0.5f * float(size_z);
					if (Square(cx) + Square(cz) <= Square(radius))
						ioObject.SetVoxel(x, y, z, true);
				}
	}
	else
	{
		// Radius of the largest shape that fits in the bounding box
		float radius_sq = Square(0.5f * size_x * shape->GetVoxelSize());

		for (int z = 0; z < size_z; ++z)
			for (int y = 0; y < size_y; ++y)
				for (int x = 0; x < size_x; ++x)
				{
					// Center of the voxel relative to the center of the shape
					Vec3 center = shape->GetVoxelCenter(x, y, z);

					// Cylinder is aligned with the local Y axis
					bool solid = inType == EShapeType::Sphere
						? center.LengthSq() <= radius_sq
						: Vec3(center.GetX(), 0, center.GetZ()).LengthSq() <= radius_sq;

					if (solid)
						ioObject.SetVoxel(x, y, z, true);
				}
	}
}

int VoxelShapeTest::RemoveVoxelsBelow(VoxelObject &ioObject, int inHeight)
{
	const VoxelShape *shape = ioObject.GetVoxelShape();

	int removed = 0;
	for (int z = 0; z < ioObject.mSizeZ; ++z)
		for (int y = 0; y < inHeight; ++y)
			for (int x = 0; x < ioObject.mSizeX; ++x)
			{
				if (ioObject.GetVoxel(x, y, z))
				{
					ioObject.SetVoxel(x, y, z, false);
					++removed;
				}
			}

	// The voxel data changed, refresh the cached properties (the solid voxel count / extent, the volume and
	// the mass properties and the debug drawing). Recalculate does the same linear scan that the shape already
	// did, so it is much cheaper than rebuilding the shape and recreating the body. Note that this is not
	// needed for the collision result to be correct: the narrow phase reads the voxel array on every query.
	shape->Recalculate();

	return removed;
}

void VoxelShapeTest::GetInitialCamera(CameraState &ioState) const
{
	const float u = sGetObjectSize();
	ioState.mPos = RVec3(0, 3.5f * u, -11.0f * u);
	ioState.mForward = Vec3(0, -0.2f, 1.0f).Normalized();
}

void VoxelShapeTest::PrePhysicsUpdate([[maybe_unused]] const PreUpdateParams &inParams)
{
	// Start the timer for the physics step, it is stopped in PostPhysicsUpdate
	mStepStart = chrono::high_resolution_clock::now();
}

void VoxelShapeTest::PostPhysicsUpdate([[maybe_unused]] float inDeltaTime)
{
	// Time that PhysicsSystem::Update took. Note that this includes the time spent waiting for the
	// simulation jobs, so it is the cost that the frame actually pays for the simulation.
	chrono::microseconds duration = chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - mStepStart);
	mStepTime = 1.0e-6f * float(duration.count());
}

String VoxelShapeTest::GetStatusString() const
{
	String result;

	for (const VoxelObject *object : mObjects)
	{
		const VoxelShape *shape = object->GetVoxelShape();
		MassProperties mass_properties = shape->GetMassProperties();
		result += StringFormat("%s: %d / %d voxels, %g m^3, %g kg\n",
			object->mName.c_str(),
			shape->GetNumSolidVoxels(),
			object->mSizeX * object->mSizeY * object->mSizeZ,
			(double)shape->GetVolume(),
			(double)mass_properties.mMass);

		// A voxel shape has no derived geometry: the narrow phase walks the solid voxels in the overlap of the
		// two shapes and collides each of them as its own small box. So the cost driver is the number of solid
		// voxels in the overlap, not the size of the grid. GetSubShapeIDBitsRecursive() is the number of bits
		// that a sub shape ID uses to encode the voxel coordinate that was hit (which is what a destruction
		// system needs to know which cells to remove).
		result += StringFormat("  surface voxels: %d, sub shape id: %d bits\n",
			shape->GetNumSurfaceVoxels(), shape->GetSubShapeIDBitsRecursive());
	}

	// Cost of the last simulation step (paused steps are not counted as they don't call PostPhysicsUpdate).
	// This is the number to watch while optimizing the collision queries (every voxel in the overlap of two
	// shapes runs a convex vs convex query, see the note in VoxelShape.h).
	if (mStepTime >= 0.0f)
		result += StringFormat("\nPhysicsSystem::Update: %.2f ms\n", (double)(1000.0f * mStepTime));
	result += "Hold the space bar or the right mouse button to drag a body around.\n";

	return result;
}

void VoxelShapeTest::CreateSettingsMenu(DebugUI *inUI, UIElement *inSubMenu)
{
	inUI->CreateTextButton(inSubMenu, "Select Voxel Size", [this, inUI]() {
		UIElement *voxel_size = inUI->CreateMenu();
		for (uint i = 0; i < size(sVoxelSizes); ++i)
			inUI->CreateTextButton(voxel_size, sVoxelSizeNames[i], [this, i]() { sVoxelSizeIndex = i; RestartTest(); });
		inUI->ShowMenu(voxel_size);
	});

	inUI->CreateTextButton(inSubMenu, "Carve Tunnel In Platform", [this]() {
		if (mPlatform != nullptr)
		{
			// Dig a tunnel straight through the platform. The voxel data is modified in place while the body is
			// alive, there is no need to rebuild the shape or to recreate the body, but Recalculate must be
			// called to refresh the cached properties and the drawing.
			// The tunnel is a real tunnel: collision is done against the individual voxels (not against a convex
			// hull of them), so a body that is small enough will fall / roll through it. Watch the bodies that
			// are stacked on top of the platform: the ones above the tunnel drop into it.
			// The tunnel is 3 x 3 cells (0.1875 m, the same size as with the old 0.1 m voxels where it was
			// 2 x 2 cells), starting 5 cells (0.3125 m) above the bottom of the platform and centered in x.
			const VoxelShape *shape = mPlatform->GetVoxelShape();
			for (int z = 0; z < mPlatform->mSizeZ; ++z)
				for (int y = 5; y <= 7; ++y)
					for (int x = 6; x <= 8; ++x)
						mPlatform->SetVoxel(x, y, z, false);
			shape->Recalculate();
		}
	});

	inUI->CreateTextButton(inSubMenu, "Remove Bottom Layer Of Platform", [this]() {
		if (mPlatform != nullptr)
			RemoveVoxelsBelow(*mPlatform, 2);			// 2 cells = 0.125 m, the old bottom layer was 0.1 m
	});

	// Fill the window of the wall: the voxel data that was concave becomes convex again, so the ball that just
	// fell through the window now rests on the wall. This shows that editing the voxels is enough to change the
	// collision behavior, there is no need to rebuild the shape or to recreate the body.
	inUI->CreateTextButton(inSubMenu, "Fill Window In Wall", [this]() {
		if (mWall != nullptr)
		{
			const VoxelShape *shape = mWall->GetVoxelShape();
			int hole_min = cNumVoxels / 2 - cWindowHalfSize;
			int hole_max = cNumVoxels / 2 + cWindowHalfSize;
			for (int z = hole_min; z < hole_max; ++z)
				for (int y = 0; y < mWall->mSizeY; ++y)
					for (int x = hole_min; x < hole_max; ++x)
						mWall->SetVoxel(x, y, z, true);
			shape->Recalculate();
		}
	});

	inUI->CreateTextButton(inSubMenu, "Reset", [this]() {
		RestartTest();
	});
}
