namespace Mel.Net
{
    public class MelFeatureExtractor: IDisposable
    {
        private int _disposeCount = 0;
        private IntPtr _featureExtractor;
        private readonly float _preemphasis;

        public unsafe MelFeatureExtractor(
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
            bool log,
            float preemphasis
            )
        {
            _featureExtractor = new IntPtr(NativeLibMelInterop.create_feature_extractor(
                sampleRate,
                winLen,
                n_fft,
                n_hop,
                window,
                power,
                n_mels,
                f_min,
                f_max,
                normalize,
                log));
            _preemphasis = preemphasis;
        }

        public unsafe void Dispose()
        {
            if (Interlocked.Increment(ref _disposeCount) == 1)
            {
                NativeLibMelInterop.delete_feature_extractor(_featureExtractor.ToPointer());
            }
        }

        public unsafe float[] ExtractFeature(float []data)
        {
            int expectedSize = NativeLibMelInterop.estimate_buffer_size(_featureExtractor.ToPointer(), data.Length);
            float[] result = new float[expectedSize];
            fixed(float *input = data)
            {
                fixed (float *output = result)
                {
                    int count = NativeLibMelInterop.extract_features(_featureExtractor.ToPointer(),
                        _preemphasis,
                        input,
                        data.Length,
                        output,
                        result.Length);
                    if (count < 0)
                        throw new OutOfMemoryException("No enough buffer size to store mel feature");
                }
            }
            return result;
        }
    }
}