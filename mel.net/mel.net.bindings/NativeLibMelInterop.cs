using System.Runtime.InteropServices;
using System.Security;

namespace Mel.Net
{

	[SuppressUnmanagedCodeSecurity]
	internal sealed class NativeLibMelInterop
	{
		private const string _lib = "mel.c";

		[DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
		public static extern unsafe void* create_feature_extractor(
			int sampleRate,
			int winLen,
			int n_fft,
			int n_hop,
			string window,
			float power,
			int n_mels,
			float f_min,
			float f_max,
			NormalizeType normalize,
			bool log
		);

		[DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
		public static extern unsafe void delete_feature_extractor(void* extractor);

		[DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
		public static extern unsafe int extract_features(
				void* extractor,
				float preemphasisK,
				float* samples,
				int sample_count,
				float* destBuffer,
				int buffer_size_samples);

		[DllImport(_lib, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
		public static extern unsafe int estimate_buffer_size(
			void* extractor,
			int sample_count);
	}

}