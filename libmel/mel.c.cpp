// mel.c.cpp : Defines the entry point for the application.
//

#include "mel.c.h"
#include "libmel.h"
#include "melFeatureExtractor.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

std::vector<float> ReadBinaryFile(const std::string& name)
{
	std::ifstream input(name, std::ios::binary);
	input.seekg(0, std::ios::end);
	const size_t num_elements = input.tellg() / sizeof(float);
	input.seekg(0, std::ios::beg);
	std::vector<float> data(num_elements);
	// copies all data into buffer
	input.read(reinterpret_cast<char*>(&data[0]), num_elements * sizeof(float));
	return data;
}

int main()
{
	auto input = ReadBinaryFile("test.f32");
	auto expected = ReadBinaryFile("out.f32");
	void* extractor = create_feature_extractor(16000, 400, 512, 160, "hann", 2.0f, 80, 0.0f, 8000.0f, PerFeature, true);
	std::vector<float> result(estimate_buffer_size(extractor, input.size()));
	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;
	auto t1 = high_resolution_clock::now();
	for (int i = 0; i < 1000; i++)
	{
		int samples = extract_features(extractor, 0.97f, input.data(), input.size(), result.data(), result.size());

		if (samples < 0)
		{
			std::cout << "No enought memory" << std::endl;
			exit(-1);
		}
	}
	auto t2 = high_resolution_clock::now();
	std::cout << (t2-t1).count()*(1E-9)<< "s/run\n";

	delete_feature_extractor(extractor);
	for (auto i = 0; i < result.size(); ++i)
	{
		if (fabs(result[i]-expected[i]) > 1E-4)
			std::cout << i << " " << expected[i] << " real: " << result[i] << std::endl;
	}
}