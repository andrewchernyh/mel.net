#include "libmel.h"
#include "melFeatureExtractor.h"
#include <vector>

Vectorf preemphasis(const float *samples, int samples_count, float coeff)
{
	Vectorf result(samples_count);
	result[0] = samples[0];
	for (size_t i = 1; i < samples_count; i++)
	{
		result[i] = samples[i] - coeff * samples[i - 1];
	}
	return result;
}

ADDExport  void* _cdecl create_feature_extractor(
	int sampleRate,
	int winlen,
	int n_fft,
	int n_hop,
	const char *window,
	float power,
	int n_mels,
	float f_min,
	float f_max,
	NormalizeType normalize,
	bool log)
{
	//no throw
	if (winlen > n_fft)
		winlen = n_fft;

	auto extractor = new MelFeatureExtractor(sampleRate,
		winlen,
		n_fft,
		n_hop,
		window,
		true,
		"constant",
		power,
		n_mels,
		f_min,
		f_max,
		normalize,
		log);
	return extractor;
}


ADDExport void _cdecl delete_feature_extractor(
	void* extractor
) {
	delete static_cast<MelFeatureExtractor *>(extractor);
}

ADDExport size_t _cdecl extract_features(
	void* extractor,
	float preemphasisK,
	const float* samples,
	size_t	   sample_count,
	float* destBuffer,
	size_t	   buffer_size_samples
)
{
	auto e = static_cast<MelFeatureExtractor*>(extractor);
	if (e->estimate_output_buffer_size(sample_count) < buffer_size_samples)
		return -1;
	auto input = preemphasis(samples, sample_count, preemphasisK);
	auto output = e->Extract(input);
	memcpy(destBuffer, output.data(), sizeof(float) * output.size());
	return output.cols();
}

ADDExport size_t _cdecl estimate_buffer_size(void* extractor, size_t sample_count)
{
	auto e = static_cast<MelFeatureExtractor*>(extractor);
	return e->estimate_output_buffer_size(sample_count);
}