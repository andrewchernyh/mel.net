#ifndef __LIBMEL_H__
#define __LIBMEL_H__

#define _cdecl
#ifdef __declspec(dllexport)
#define ADDExport __declspec (dllexport)
#else
#define ADDExport
#endif  // !__declspec (dllexport)

typedef enum
{
	None = 0,
	PerFeature = 1
} NormalizeType;

#ifdef __cplusplus
extern "C" {
#endif
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
		bool log);

		ADDExport void _cdecl delete_feature_extractor(
			void* extractor
		);

		ADDExport size_t _cdecl extract_features(
			void* extractor,
			float preemphasisK,
			const float* samples,
			size_t	   sample_count,
			float* destBuffer,
			size_t	   buffer_size_samples
		);

		ADDExport size_t _cdecl estimate_buffer_size(void* extractor, size_t sample_count);
#ifdef __cplusplus
}
#endif



#endif //__LIBMEL_H__