#include <vector>
#include "../../mesh/Logger.h"

namespace core {
	class AITools {
	public:
		static void FPS(const std::vector<std::vector<float>>& inputPts, const int sampleSize, std::vector<int>& sampleVIds);
	};
}
