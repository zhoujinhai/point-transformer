#include "Tools.h"

#include "Point.h"
#include "KDLineTree.h"

namespace core
{
	int CalKdtreeHeight(int pointSize, int targetBucketSize = 350) {
		// 350 is experience
		return static_cast<int>(std::log2(pointSize / targetBucketSize));
	}

	void AITools::FPS(const std::vector<std::vector<float>>& inputPts, const int sampleSize, std::vector<int>& sampleVIds) 
	{
		if (inputPts.size() < 1 || inputPts[0].size() < 3) {
			HGAPI_LOG("AITools::FPS", "inputPts is not avalid!");
			return;
		}

		const int pointSize = inputPts.size(); 
		//copy to array
		Point* points = (Point*)malloc(pointSize * sizeof(Point));
		for (int idx = 0; idx < pointSize; idx++) { 
			points[idx] = Point(inputPts[idx][0], inputPts[idx][1], inputPts[idx][2], 1 << 30, idx);
		} 
		Point* samplePoints = (Point*)malloc(sampleSize * sizeof(Point));

		Point init_point = points[0];
		samplePoints[0] = init_point;
		int kdtreeHeight = CalKdtreeHeight(pointSize);
		KDLineTree tree = KDLineTree(points, pointSize, kdtreeHeight, samplePoints);
		tree.buildKDtree();
		tree.init(init_point);
		tree.sample(sampleSize);
		for (int i = 0; i < sampleSize; ++i) {
			sampleVIds.push_back(samplePoints[i].id);
		}
		free(samplePoints);
		free(points);
		return;
	}
}
