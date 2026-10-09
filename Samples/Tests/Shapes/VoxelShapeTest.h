// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Tests/Test.h>
#include <Jolt/Physics/Collision/Shape/VoxelShape.h>

JPH_SUPPRESS_WARNINGS_STD_BEGIN
#include <chrono>
JPH_SUPPRESS_WARNINGS_STD_END

/// Test scene for VoxelShape: a rigid body collider made out of a grid of cubic voxels.
///
	/// The scene exercises all collision paths that a voxel shape takes part in:
	/// - A voxel cube / sphere / cylinder dropping on the box floor (voxel vs convex shape)
	/// - A voxel cube dropping on a static voxel platform and a stack of voxel cubes (voxel vs voxel)
	/// - A voxel cube dropping on a mesh shape (voxel vs mesh)
	/// - A voxel sphere dropping on a height field (voxel vs height field)
	/// - A ball dropping on a wall with a window in it: it falls through the window instead of landing on the
	///   wall, which shows that a voxel shape collides as the voxels are drawn and not as their convex hull
///
/// Useful keys while running the scene:
/// - 'H' toggles drawing of the shapes, 'Alt+W' draws them in wireframe (handy to see the individual voxels)
/// - 'J' or the "Draw Shape Color" combo box in the Drawing Options menu colors the voxels by body
///
/// The optional settings menu (top left) allows you to change the voxel size, carve voxels out of the platform
/// while the simulation is running and show the current statistics of every voxel shape.
class VoxelShapeTest : public Test
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, VoxelShapeTest)

	// Destructor
	virtual						~VoxelShapeTest() override;

	// See: Test
	virtual void				Initialize() override;
	virtual void				PrePhysicsUpdate(const PreUpdateParams &inParams) override;
	virtual void				PostPhysicsUpdate(float inDeltaTime) override;
	virtual String				GetStatusString() const override;

	// Override to specify the initial camera state (local to GetCameraPivot)
	virtual void				GetInitialCamera(CameraState &ioState) const override;

	// Optional settings menu
	virtual bool				HasSettingsMenu() const override							{ return true; }
	virtual void				CreateSettingsMenu(DebugUI *inUI, UIElement *inSubMenu) override;

private:
	/// Shape of the voxel grid that we fill in
	enum class EShapeType
	{
		Cube,																				///< All voxels filled
		Sphere,																				///< Voxels whose center is within the inscribed sphere
		Cylinder,																			///< Voxels whose center is within the inscribed cylinder (aligned with the local Y axis)
		Window,																				///< All voxels filled except a rectangular shaft through the middle (concave)
	};

	/// A voxel shape plus the voxel data that it points to.
	/// VoxelShape does not own its voxel data, so we have to keep it alive for as long as the shape is alive.
	struct VoxelObject
	{
		Array<uint8>			mVoxels;													///< Voxel data, 0 = empty, everything else = solid
		RefConst<Shape>			mShape;														///< The shape that references mVoxels
		BodyID					mBodyID;													///< The body that uses the shape
		String					mName;														///< Name used for the status string

		const VoxelShape *		GetVoxelShape() const										{ return static_cast<const VoxelShape *>(mShape.GetPtr()); }
	};

	/// Create a voxel object (cNumVoxels^3 voxels) and a body for it. The returned object is owned by this class.
	VoxelObject *				CreateVoxelObject(EShapeType inType, const char *inName, const RVec3 &inPosition, EMotionType inMotionType);

	/// Fill the voxel data of an object with the requested shape
	void						FillVoxels(VoxelObject &ioObject, EShapeType inType);

	/// Remove all voxels below inHeight (in voxel coordinates), returns the number of removed voxels
	int							RemoveVoxelsBelow(VoxelObject &ioObject, int inHeight);

	/// Size of one voxel in meters for the currently selected option
	static float				sGetVoxelSize()												{ return sVoxelSizes[sVoxelSizeIndex]; }

	/// Size of the bounding box of a voxel object in meters
	static float				sGetObjectSize()											{ return cNumVoxels * sGetVoxelSize(); }

	/// Anything holding voxel data (we own these)
	Array<VoxelObject *>		mObjects;

	/// Shortcut to the static platform so that we can modify its voxels while the simulation is running
	VoxelObject *				mPlatform = nullptr;

	/// Shortcut to the static wall with the window in it (see EShapeType::Window)
	VoxelObject *				mWall = nullptr;

	/// Number of voxels along every axis (the test uses a 10 x 10 x 10 grid)
	static constexpr int		cNumVoxels = 10;

	/// Density of the voxels in kg / m^3
	static constexpr float		cDensity = 1000.0f;

	/// Available voxel sizes in meters (the default of 0.1 gives a 1 x 1 x 1 meter bounding box)
	inline static const float	sVoxelSizes[] = { 0.05f, 0.1f, 0.2f };

	/// Index into sVoxelSizes of the currently selected voxel size
	inline static int			sVoxelSizeIndex = 1;

	/// Time the physics system spends in one step (in seconds), measured around PhysicsSystem::Update.
	/// Shown in the status string so that the effect of collision optimizations can be judged while running.
	std::chrono::high_resolution_clock::time_point mStepStart;
	float						mStepTime = -1.0f;
};
