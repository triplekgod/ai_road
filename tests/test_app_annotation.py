import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import app
from annotate import Annotator, MaskEditor


class MaskEditorTests(unittest.TestCase):
    def test_multiple_polygons_preserve_separate_regions(self):
        editor = MaskEditor(np.zeros((32, 48), np.uint8))
        self.assertTrue(editor.polygon([(2, 2), (12, 2), (12, 12), (2, 12)]))
        self.assertTrue(editor.polygon([(25, 17), (40, 17), (40, 28), (25, 28)]))
        self.assertEqual(cv2.connectedComponents(editor.mask)[0] - 1, 2)
        self.assertEqual(editor.mask[7, 7], 255)
        self.assertEqual(editor.mask[20, 30], 255)
        self.assertEqual(editor.mask[20, 17], 0)
        self.assertEqual(set(np.unique(editor.mask)), {0, 255})

    def test_subtraction_black_and_undo_restore_previous_masks(self):
        editor = MaskEditor(np.zeros((32, 32), np.uint8))
        initial = editor.mask.copy()
        editor.polygon([(2, 2), (29, 2), (29, 29), (2, 29)])
        added = editor.mask.copy()
        editor.polygon([(10, 10), (20, 10), (20, 20), (10, 20)], subtract=True)
        subtracted = editor.mask.copy()
        self.assertFalse(np.any(subtracted[10:21, 10:21]))
        self.assertEqual(subtracted[5, 5], 255)
        editor.black()
        self.assertFalse(np.any(editor.mask))
        for expected in (subtracted, added, initial):
            editor.undo()
            np.testing.assert_array_equal(editor.mask, expected)
        editor.undo()
        np.testing.assert_array_equal(editor.mask, initial)

    def test_incomplete_polygon_does_not_modify_or_add_history(self):
        editor = MaskEditor(np.ones((16, 16), np.uint8))
        before = editor.mask.copy()
        self.assertFalse(editor.polygon([(1, 1), (10, 10)], subtract=True))
        self.assertEqual(editor.history, [])
        np.testing.assert_array_equal(editor.mask, before)

    def test_existing_mask_overwrite_needs_confirmation_then_auto_advances(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            images, masks = base / "images", base / "masks"
            images.mkdir()
            masks.mkdir()
            for name in ("a", "b"):
                self.assertTrue(cv2.imwrite(str(images / f"{name}.bmp"), np.zeros((24, 32, 3), np.uint8)))
            mask_path = masks / "a_mask.bmp"
            self.assertTrue(cv2.imwrite(str(mask_path), np.full((24, 32), 255, np.uint8)))
            before = mask_path.read_bytes()
            with patch("annotate.cv2.namedWindow"), patch("annotate.cv2.setMouseCallback"):
                annotator = Annotator(images, masks)
            self.assertTrue(np.all(annotator.editor.mask == 255))
            annotator.editor.black()
            annotator.changed()
            annotator.save()
            self.assertEqual(mask_path.read_bytes(), before)
            self.assertTrue(annotator.overwrite_armed)
            self.assertEqual(annotator.index, 0)
            with contextlib.redirect_stdout(io.StringIO()):
                annotator.save()
            self.assertFalse(np.any(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)))
            self.assertEqual(annotator.index, 1)
            self.assertFalse(annotator.dirty)

    def test_navigation_requires_confirmation_for_unsaved_work(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            images = base / "images"
            images.mkdir()
            for name in ("a", "b"):
                cv2.imwrite(str(images / f"{name}.bmp"), np.zeros((24, 32, 3), np.uint8))
            with patch("annotate.cv2.namedWindow"), patch("annotate.cv2.setMouseCallback"):
                annotator = Annotator(images, base / "masks")
            annotator.points = [(2, 2), (20, 2), (20, 20)]
            annotator.commit()
            annotator.navigate(1)
            self.assertEqual(annotator.index, 0)
            self.assertTrue(annotator.dirty)
            annotator.navigate(1)
            self.assertEqual(annotator.index, 1)
            self.assertFalse(annotator.dirty)

    def test_editing_after_quit_warning_requires_fresh_confirmation(self):
        annotator = Annotator.__new__(Annotator)
        annotator.editor = MaskEditor(np.ones((16, 16), np.uint8))
        annotator.points = []
        annotator.dirty, annotator.overwrite_armed = True, False
        annotator.pending_navigation, annotator.notice = None, ""
        with patch.object(annotator, "render"), \
                patch("annotate.cv2.waitKey", side_effect=[ord("q"), ord("b"), ord("q"), ord("q")]) as keyboard, \
                patch("annotate.cv2.destroyAllWindows"):
            annotator.run()
        self.assertEqual(keyboard.call_count, 4, "A new edit must cancel the previous quit confirmation")


class ConsoleAppTests(unittest.TestCase):
    def test_new_training_includes_chosen_model_and_preprocessing(self):
        answers = ["splits.json", "12", "4", "models/best.pth", "fast_scnn_lite", "224", "128", "0.25"]
        with patch("app.ask", side_effect=answers), patch("app.run") as run:
            app.train()
        script, arguments = run.call_args.args
        self.assertEqual(script, "train.py")
        pairs = dict(zip(arguments[::2], arguments[1::2]))
        self.assertEqual(pairs["--architecture"], "fast_scnn_lite")
        self.assertEqual((pairs["--input-width"], pairs["--input-height"], pairs["--roi-top"]), ("224", "128", "0.25"))
        self.assertNotIn("--resume", pairs)
        self.assertNotIn("--finetune", pairs)

    def test_resume_and_finetune_pass_distinct_flags_without_overriding_metadata(self):
        for mode in ("resume", "finetune"):
            with self.subTest(mode=mode):
                answers = ["splits.json", "18", "2", "models/best.pth", "models/previous.last.pth"]
                with patch("app.ask", side_effect=answers), patch("app.run") as run:
                    app.train(mode)
                script, arguments = run.call_args.args
                self.assertEqual(script, "train.py")
                pairs = dict(zip(arguments[::2], arguments[1::2]))
                self.assertEqual(pairs["--" + mode], "models/previous.last.pth")
                self.assertNotIn("--finetune" if mode == "resume" else "--resume", pairs)
                self.assertNotIn("--architecture", pairs)
                self.assertNotIn("--input-width", pairs)
                self.assertNotIn("--roi-top", pairs)

    def test_menu_starts_and_exits_with_zero_without_gui(self):
        result = subprocess.run([sys.executable, str(Path(app.__file__).resolve())], input="0\n",
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("resume", result.stdout)
        self.assertIn("finetune", result.stdout)


if __name__ == "__main__":
    unittest.main()
