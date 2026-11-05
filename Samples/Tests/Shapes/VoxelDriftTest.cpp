// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <TestFramework.h>

#include <Tests/Shapes/VoxelDriftTest.h>
#include <Jolt/Core/StringTools.h>
JPH_SUPPRESS_WARNINGS_STD_BEGIN
#include <cstdint>
JPH_SUPPRESS_WARNINGS_STD_END
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Renderer/DebugRenderer.h>
#include <Utils/DebugRendererSP.h>
#include <Utils/Log.h>
#include <Layers.h>

JPH_IMPLEMENT_RTTI_VIRTUAL(VoxelDriftTest)
{
	JPH_ADD_BASE_CLASS(VoxelDriftTest, Test)
}

VoxelDriftTest::~VoxelDriftTest()
{
	// The shapes reference the voxel data that we own, so remove the bodies before throwing the voxel data away.
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

bool VoxelDriftTest::LoadTerrainDump(const char *inPath)
{
	// 与 UnitTests 的 VoxelShapeUnityReplay 共用同一份格式（Unity 菜单导出）：
	//   magic 'VDMP' | version 1 | float voxel_size
	//   int32 origin.x/y/z, size.x/y/z | float terrain_node_pos[3]
	//   uint8 occupancy[size.x * size.y * size.z]（x 最快、然后 y、最后 z）
	//   int32 block_cells.x/y/z | float block_pos[3] | float block_rot[4] | float block_vel[3]
	std::ifstream f(inPath, std::ios::binary);
	if (!f.good())
	{
		Trace("VoxelDrift: 读不到地形 dump (%s)", inPath);
		return false;
	}

	int32_t magic = 0, version = 0;
	f.read(reinterpret_cast<char *>(&magic), 4);
	f.read(reinterpret_cast<char *>(&version), 4);
	if (magic != 0x504D4456 || version != 1)
	{
		Trace("VoxelDrift: dump 格式不对 magic=%08x version=%d", (uint32_t)magic, version);
		return false;
	}

	float vs = 0.0f;
	int32_t ox = 0, oy = 0, oz = 0, nx = 0, ny = 0, nz = 0;
	float terrain_pos[3];
	f.read(reinterpret_cast<char *>(&vs), 4);
	f.read(reinterpret_cast<char *>(&ox), 4);
	f.read(reinterpret_cast<char *>(&oy), 4);
	f.read(reinterpret_cast<char *>(&oz), 4);
	f.read(reinterpret_cast<char *>(&nx), 4);
	f.read(reinterpret_cast<char *>(&ny), 4);
	f.read(reinterpret_cast<char *>(&nz), 4);
	f.read(reinterpret_cast<char *>(terrain_pos), 12);
	if (!f.good() || nx <= 0 || ny <= 0 || nz <= 0)
		return false;

	std::vector<uint8_t> occ(size_t(nx) * ny * nz, 0);
	f.read(reinterpret_cast<char *>(occ.data()), std::streamsize(occ.size()));

	int32_t gx = 0, gy = 0, gz = 0;
	float bpos[3] = { 0, 0, 0 }, brot[4] = { 0, 0, 0, 1 }, bvel[3] = { 0, 0, 0 };
	f.read(reinterpret_cast<char *>(&gx), 4);
	f.read(reinterpret_cast<char *>(&gy), 4);
	f.read(reinterpret_cast<char *>(&gz), 4);
	f.read(reinterpret_cast<char *>(bpos), 12);
	f.read(reinterpret_cast<char *>(brot), 16);
	f.read(reinterpret_cast<char *>(bvel), 12);
	f.close();

	mDumpVoxelSize = vs;
	mDumpBlockCells = max(1, gx);
	mDumpBlockPos = RVec3(bpos[0], bpos[1], bpos[2]);

	// ---- 按 Unity 的布局建地形：每个 16^3 体素 chunk 一个静态 body ----
	// 世界坐标：格 i 占 [origin + i, origin + i + 1) * vs，chunk 中心 = (origin + chunk_base + 8) * vs
	// 全空 chunk 不建 body（与 Unity 的 EmptyNoBody 一致）。
	int num_chunks = 0;
	for (int cz = 0; cz + cChunkCells <= nz; cz += cChunkCells)
		for (int cy = 0; cy + cChunkCells <= ny; cy += cChunkCells)
			for (int cx = 0; cx + cChunkCells <= nx; cx += cChunkCells)
			{
				VoxelObject *object = new VoxelObject();
				object->mSizeX = object->mSizeY = object->mSizeZ = cChunkCells;
				object->mTexelsX = (object->mSizeX + 3) >> 2;
				object->mTexelsY = (object->mSizeY + 3) >> 2;
				object->mTexelPlane = object->mTexelsX * object->mTexelsY;
				object->mOccupancy.resize(size_t(object->mTexelPlane) * ((object->mSizeZ + 3) >> 2), 0);

				bool any = false;
				for (int z = 0; z < cChunkCells; ++z)
					for (int y = 0; y < cChunkCells; ++y)
						for (int x = 0; x < cChunkCells; ++x)
							if (occ[size_t(cx + x) + size_t(nx) * ((cy + y) + size_t(ny) * (cz + z))] != 0)
							{
								object->SetVoxel(x, y, z);
								any = true;
							}

				if (!any)
				{
					delete object;
					continue;
				}

				object->mShape = new VoxelShape(object->mSizeX, object->mSizeY, object->mSizeZ, vs, object->mOccupancy.data());

				// 区块侧壁接的是隔壁区块的实心地形，不是真边界
				const_cast<VoxelShape *>(static_cast<const VoxelShape *>(object->mShape.GetPtr()))->SetBorderContinues(true);

				RVec3 center = RVec3(terrain_pos[0] + (float(ox + cx + cChunkCells / 2)) * vs,
						terrain_pos[1] + (float(oy + cy + cChunkCells / 2)) * vs,
						terrain_pos[2] + (float(oz + cz + cChunkCells / 2)) * vs);
				object->mBodyID = mBodyInterface->CreateAndAddBody(BodyCreationSettings(object->mShape, center, Quat::sIdentity(), EMotionType::Static, Layers::NON_MOVING), EActivation::DontActivate);

				mObjects.push_back(object);
				++num_chunks;
			}

	Trace("VoxelDrift: 地形 dump 已加载 %.4f m 格 | 窗口 %dx%dx%d | chunk %d 个 | 方块 %d^3 @ (%.3f, %.3f, %.3f)",
		(double)vs, nx, ny, nz, num_chunks, mDumpBlockCells, bpos[0], bpos[1], bpos[2]);
	return true;
}

VoxelDriftTest::VoxelObject *VoxelDriftTest::CreateBlock(const RVec3 &inPosition, float inVoxelSize, int inVoxelCount, FallingBlock &ioBlock)
{
	VoxelObject *object = new VoxelObject();
	object->mSizeX = object->mSizeY = object->mSizeZ = inVoxelCount;
	object->mTexelsX = (object->mSizeX + 3) >> 2;
	object->mTexelsY = (object->mSizeY + 3) >> 2;
	object->mTexelPlane = object->mTexelsX * object->mTexelsY;
	object->mOccupancy.resize(size_t(object->mTexelPlane) * ((object->mSizeZ + 3) >> 2), 0);

	for (int z = 0; z < object->mSizeZ; ++z)
		for (int y = 0; y < object->mSizeY; ++y)
			for (int x = 0; x < object->mSizeX; ++x)
				object->SetVoxel(x, y, z);

	object->mShape = new VoxelShape(object->mSizeX, object->mSizeY, object->mSizeZ, inVoxelSize, object->mOccupancy.data());
	object->mBodyID = mBodyInterface->CreateAndAddBody(BodyCreationSettings(object->mShape, inPosition, Quat::sIdentity(), EMotionType::Dynamic, Layers::MOVING), EActivation::Activate);

	ioBlock.mID = object->mBodyID;
	ioBlock.mStart = ioBlock.mPrev = inPosition;

	mObjects.push_back(object);
	return object;
}

void VoxelDriftTest::Initialize()
{
	// 兜底地面（掉出去的东西最终会落在它上面，方便区分"从缝里穿下去"和"什么都没建"）
	CreateFloor(300.0f);

	bool loaded = LoadTerrainDump("F:/GameProject/voxel/Tools/terrain_dump.bin");
	if (!loaded)
		Trace("VoxelDrift: dump 未加载 —— 场景里只有兜底地面，请先在 Unity 里导出 F:/GameProject/voxel/Tools/terrain_dump.bin");

	// ---- 三个 8^3 体素（= 0.5 m）方块，就是 dump 里导出的那个 ----
	mBlocks[0].mName = "dump pose (red)";
	mBlocks[1].mName = "same XZ +2 m (green)";
	mBlocks[2].mName = "offset +3 m (yellow)";
	CreateBlock(mDumpBlockPos, mDumpVoxelSize, mDumpBlockCells, mBlocks[0]);
	CreateBlock(RVec3(float(mDumpBlockPos.GetX()), float(mDumpBlockPos.GetY()) + 2.0f, float(mDumpBlockPos.GetZ())), mDumpVoxelSize, mDumpBlockCells, mBlocks[1]);
	CreateBlock(RVec3(float(mDumpBlockPos.GetX()) + 0.75f, float(mDumpBlockPos.GetY()) + 3.0f, float(mDumpBlockPos.GetZ()) + 0.75f), mDumpVoxelSize, mDumpBlockCells, mBlocks[2]);

	// 命令行运行 Samples.exe 时 Trace 只走 OutputDebugString，这里同时落一份到文件
	mLog.open("F:/GameProject/voxel/Tools/samples_drift.log", std::ios::out | std::ios::trunc);
}

void VoxelDriftTest::PostPhysicsUpdate([[maybe_unused]] float inDeltaTime)
{
	if (DebugRenderer::sInstance == nullptr)
		return;

	const Color colors[3] = { Color::sRed, Color::sGreen, Color::sYellow };
	++mFrameCount;

	for (int i = 0; i < 3; ++i)
	{
		FallingBlock &b = mBlocks[i];
		if (b.mID.IsInvalid())
			continue;

		RVec3 p = mBodyInterface->GetPosition(b.mID);
		Vec3 v = mBodyInterface->GetLinearVelocity(b.mID);

		// 抖动窗口统计：y 峰峰值 / |vy| 峰值 / vy 换向次数
		b.mWinMinY = min(b.mWinMinY, float(p.GetY()));
		b.mWinMaxY = max(b.mWinMaxY, float(p.GetY()));
		b.mWinMaxVy = max(b.mWinMaxVy, abs(v.GetY()));
		{
			int sg = v.GetY() > 0.01f? 1 : (v.GetY() < -0.01f? -1 : 0);
			if (sg != 0)
			{
				if (b.mLastVySign != 0 && sg != b.mLastVySign)
					++b.mWinFlips;
				b.mLastVySign = sg;
			}
		}

		// 轨迹：上一帧 -> 这一帧。横向的线段 = 横着滑走，竖直向下的长线 = 穿透掉下去。
		DrawLineSP(DebugRenderer::sInstance, b.mPrev, p, colors[i]);
		b.mPrev = p;

		float drift = Vec3(float(p.GetX()) - float(b.mStart.GetX()), 0.0f, float(p.GetZ()) - float(b.mStart.GetZ())).Length();
		b.mMaxDrift = max(b.mMaxDrift, drift);
		b.mMinY = min(b.mMinY, float(p.GetY()));

		// 起点
		DrawMarkerSP(DebugRenderer::sInstance, b.mStart, colors[i], 0.06f);

		// 每秒打一行（VS 输出窗口可见 + 落文件），不看画面也能对数值
		float drop = float(b.mStart.GetY()) - float(p.GetY());
		bool fell = float(p.GetY()) < -1.0f;
		if (mFrameCount % 60 == 0 || (fell && !b.mReportedFall))
		{
			b.mReportedFall = b.mReportedFall || fell;
			char line[256];
			snprintf(line, sizeof(line), "%-22s frame %4d | pos (%.3f, %.3f, %.3f) | drift %.3f m | drop %.3f m | v %.2f m/s",
				b.mName, mFrameCount, float(p.GetX()), float(p.GetY()), float(p.GetZ()), drift, drop, v.Length());
			Trace("VoxelDrift: %s", line);
			if (mLog.is_open())
				mLog << line << std::endl;

			// 抖动：1 秒内 y 的峰峰值（米 / 格）、|vy| 峰值、竖直速度换向次数
			float pp = b.mWinMaxY - b.mWinMinY;
			char jline[256];
			snprintf(jline, sizeof(jline), "%-22s jitter 1s: y pp %.5f m (%.2f ge) | max|vy| %.3f m/s | fanxiang %d",
				b.mName, pp, pp / mDumpVoxelSize, b.mWinMaxVy, b.mWinFlips);
			Trace("VoxelDrift: %s", jline);
			if (mLog.is_open())
				mLog << jline << std::endl;

			b.mWinMinY = 1.0e9f;
			b.mWinMaxY = -1.0e9f;
			b.mWinMaxVy = 0.0f;
			b.mWinFlips = 0;
		}
	}
}

String VoxelDriftTest::GetStatusString() const
{
	if (mBodyInterface == nullptr)
		return "";

	String result;
	for (int i = 0; i < 3; ++i)
	{
		const FallingBlock &b = mBlocks[i];
		if (b.mID.IsInvalid())
			continue;

		RVec3 p = mBodyInterface->GetPosition(b.mID);
		Vec3 v = mBodyInterface->GetLinearVelocity(b.mID);
		float drift = Vec3(float(p.GetX()) - float(b.mStart.GetX()), 0.0f, float(p.GetZ()) - float(b.mStart.GetZ())).Length();

		result += StringFormat("%s: y %.2f | drift %.3f m | drop %.2f m | v %.1f m/s\n",
			b.mName, float(p.GetY()), drift, float(b.mStart.GetY()) - float(p.GetY()), v.Length());
	}
	return result;
}

void VoxelDriftTest::GetInitialCamera(CameraState &ioState) const
{
	// dump 窗口：x [-32, -16]、y [8, 16]、z [-16, 0]，方块在 (-24.7, 8.4, -9.7) 附近
	ioState.mPos = RVec3(-24.0f, 13.0f, 6.0f);
	ioState.mForward = Vec3(0.0f, -0.38f, -1.0f).Normalized();
}
