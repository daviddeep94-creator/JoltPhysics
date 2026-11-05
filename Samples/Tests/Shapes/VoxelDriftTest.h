// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Tests/Test.h>
#include <Jolt/Physics/Collision/Shape/VoxelShape.h>

JPH_SUPPRESS_WARNINGS_STD_BEGIN
#include <fstream>
JPH_SUPPRESS_WARNINGS_STD_END

/// 复现场景：**Unity 导出的真实地形（dump）上落 8 x 8 x 8 体素方块**。
///
/// 地形来自 `F:/GameProject/voxel/Tools/terrain_dump.bin`（Unity 菜单"导出地形回放数据"）：
///   - 体素 0.0625 m，窗口 256 x 128 x 256 格 = 16 x 8 x 16 m；
///   - 与 Unity 一致：**每个 16^3 体素 chunk 一个静态 VoxelShape / 一个 body**
///     （1 m 立方，约 2048 块），`SetBorderContinues(true)`（区块侧壁是假边界）；
///   - 里面是真实地形（斜坡 / 台阶 / 洞口都是原样的，不是程序生成的台阶）。
///
/// 方块：dump 里导出时的那个 —— 8 x 8 x 8 体素 = 0.5 m 立方。三个：
///   - 红：dump 位姿原地（就是 Unity 现场那一下）；
///   - 绿：同 XZ 抬高 2 m 落下来（撞击）；
///   - 黄：旁边一格抬高 3 m（换个落点看是否只在特定位置出问题）。
///
/// 运行后屏幕上会画出：
///   - 红 / 绿 / 黄折线：方块中心的运动轨迹（横向的线 = 漂移，竖直下坠的线 = 穿透）
///   - 起点标记
///   - 状态栏：各方块的位置 / 水平漂移 / 垂直位移 / 速度
///
/// 键盘：'H' 切换形状显示，Alt+W 线框，空格暂停。
class VoxelDriftTest : public Test
{
public:
	JPH_DECLARE_RTTI_VIRTUAL(JPH_NO_EXPORT, VoxelDriftTest)

	// Destructor
	virtual						~VoxelDriftTest() override;

	// See: Test
	virtual void				Initialize() override;
	virtual void				PostPhysicsUpdate(float inDeltaTime) override;
	virtual String				GetStatusString() const override;
	virtual void				GetInitialCamera(CameraState &ioState) const override;

private:
	/// A voxel grid plus the occupancy bitmap that the shape points to (VoxelShape does not own it).
	struct VoxelObject
	{
		Array<uint64>			mOccupancy;
		int						mSizeX = 0, mSizeY = 0, mSizeZ = 0;
		int						mTexelsX = 1, mTexelsY = 1, mTexelPlane = 1;
		RefConst<Shape>			mShape;
		BodyID					mBodyID;

		inline void				SetVoxel(int inX, int inY, int inZ)
		{
			uint64 &word = mOccupancy[(inX >> 2) + (inY >> 2) * mTexelsX + (inZ >> 2) * mTexelPlane];
			word |= uint64(1) << ((inX & 3) | ((inY & 3) << 2) | ((inZ & 3) << 4));
		}
	};

	/// 一个下落方块 + 它的轨迹
	struct FallingBlock
	{
		BodyID					mID;
		RVec3					mStart = RVec3::sZero();
		RVec3					mPrev = RVec3::sZero();
		float					mMaxDrift = 0.0f;
		float					mMinY = 1.0e9f;
		bool					mReportedFall = false;
		// 1 秒窗口的抖动统计（y 峰峰值 / 速度峰值 / 竖直速度换向次数）
		float					mWinMinY = 1.0e9f;
		float					mWinMaxY = -1.0e9f;
		float					mWinMaxVy = 0.0f;
		int						mWinFlips = 0;
		int						mLastVySign = 0;
		const char *			mName = "";
	};

	/// 读 Unity 导出的地形 dump，按「每 16^3 体素一个静态 chunk body」建出地形。
	/// 返回 false = 文件缺失 / 格式不对（场景里就只剩兜底地面）。
	bool						LoadTerrainDump(const char *inPath);

	/// 建一个全实心的体素方块并创建动态 body
	VoxelObject *				CreateBlock(const RVec3 &inPosition, float inVoxelSize, int inVoxelCount, FallingBlock &ioBlock);

	/// Anything holding voxel data (we own these)
	Array<VoxelObject *>		mObjects;

	/// dump 位姿 / 抬高 2 m / 旁边抬高 3 m
	FallingBlock				mBlocks[3];

	/// dump 里方块的信息（Initialize 摆放用）
	RVec3						mDumpBlockPos = RVec3::sZero();
	float						mDumpVoxelSize = 0.0625f;
	int							mDumpBlockCells = 8;

	int							mFrameCount = 0;
	std::ofstream				mLog;		// 同时写文件，命令行跑也能对数值

	// Unity 侧 TerrainPhysChunk = 16^3 体素（0.0625 m 格 ⇒ 1 m 立方）
	static constexpr int		cChunkCells = 16;
};
