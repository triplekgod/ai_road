"""Reject output aliases before loading models or touching source video."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from infer import main
from video_pipeline import run_video


class ArtifactSafetyTests(unittest.TestCase):
    def test_report_cannot_replace_input_or_encoded_video(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source.mp4', Path(directory) / 'output.mp4'
            source.write_bytes(b'original')
            output.write_bytes(b'previous result')
            for report in (source, output):
                with self.subTest(report=report), patch('video_pipeline.cv2.VideoCapture') as capture:
                    with self.assertRaisesRegex(ValueError, 'Report must not overwrite'):
                        run_video(Mock(), source, output=output, report_path=report)
                    capture.assert_not_called()
            self.assertEqual(source.read_bytes(), b'original')
            self.assertEqual(output.read_bytes(), b'previous result')

    def test_cli_outputs_cannot_replace_model_or_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'road.onnx'
            sidecar = Path(str(model) + '.json')
            model.write_bytes(b'model')
            sidecar.write_bytes(b'metadata')
            for flag in ('--output', '--report'):
                for target in (model, sidecar):
                    with self.subTest(flag=flag, target=target), patch('infer.RoadAnalyzer') as analyzer:
                        with self.assertRaisesRegex(ValueError, 'must not overwrite the model'):
                            main(['video.mp4', str(model), flag, str(target)])
                        analyzer.assert_not_called()
            self.assertEqual(model.read_bytes(), b'model')
            self.assertEqual(sidecar.read_bytes(), b'metadata')


if __name__ == '__main__':
    unittest.main()
