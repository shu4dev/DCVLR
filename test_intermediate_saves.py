"""
Test script to verify intermediate save functionality.
This creates a minimal test to ensure the save methods work correctly.
"""

import json
import tempfile
from pathlib import Path
import yaml

# Test the intermediate save methods
def test_intermediate_saves():
    print("Testing intermediate save functionality...")

    # Create a temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nUsing temporary directory: {tmpdir}")

        # Create a minimal config
        config = {
            'filtering': {
                'min_resolution': 256,
                'nsfw_threshold': 0.5,
                'phash_threshold': 8,
                'enable_watermark_check': True,
                'watermark_edge_ratio': 0.3
            },
            'binning': {
                'text_boxes_threshold': 2,
                'object_count_threshold': 5,
                'unique_objects_threshold': 3,
                'clip_similarity_threshold': 0.25,
                'use_paddle_ocr': True,
                'object_detector': 'yolo',
                'yolo_model': 'yolov8n',
                'enable_multi_gpu': False
            },
            'output': {
                'save_intermediate': True
            },
            'synthesis': {},
            'validation': {}
        }

        # Save config to temp file
        config_path = Path(tmpdir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Import pipeline (do this after config is created)
        from team1_pipeline import DataSynthesisPipeline

        # Create pipeline instance
        pipeline = DataSynthesisPipeline(
            config_path=str(config_path),
            output_dir=tmpdir,
            device='cpu'
        )

        print(f"✓ Pipeline initialized with save_intermediate={pipeline.save_intermediate}")

        # Test Stage 1 save
        print("\n--- Testing Stage 1 (Filtering) save ---")
        test_filtered_images = [
            {'path': '/path/to/image1.jpg', 'id': 'img1'},
            {'path': '/path/to/image2.jpg', 'id': 'img2'},
            {'path': '/path/to/image3.jpg', 'id': 'img3'}
        ]
        pipeline.save_stage1_results(test_filtered_images)

        stage1_dir = Path(tmpdir) / "intermediate" / "stage1_filtering"
        assert stage1_dir.exists(), "Stage 1 directory not created"
        assert (stage1_dir / "filtered_images.jsonl").exists(), "filtered_images.jsonl not created"
        assert (stage1_dir / "summary.json").exists(), "summary.json not created"

        # Verify content
        with open(stage1_dir / "summary.json") as f:
            summary = json.load(f)
            assert summary['total_filtered'] == 3, "Incorrect count in summary"
        print(f"✓ Stage 1 results saved to {stage1_dir}")

        # Test Stage 2 save
        print("\n--- Testing Stage 2 (Binning) save ---")
        test_binned_images = {
            'A': [{'path': '/path/to/image1.jpg', 'id': 'img1'}],
            'B': [{'path': '/path/to/image2.jpg', 'id': 'img2'}],
            'C': [{'path': '/path/to/image3.jpg', 'id': 'img3'}]
        }
        pipeline.save_stage2_results(test_binned_images)

        stage2_dir = Path(tmpdir) / "intermediate" / "stage2_binning"
        assert stage2_dir.exists(), "Stage 2 directory not created"
        assert (stage2_dir / "bin_A.jsonl").exists(), "bin_A.jsonl not created"
        assert (stage2_dir / "bin_B.jsonl").exists(), "bin_B.jsonl not created"
        assert (stage2_dir / "bin_C.jsonl").exists(), "bin_C.jsonl not created"
        assert (stage2_dir / "all_binned_images.jsonl").exists(), "all_binned_images.jsonl not created"
        assert (stage2_dir / "summary.json").exists(), "summary.json not created"

        # Verify content
        with open(stage2_dir / "summary.json") as f:
            summary = json.load(f)
            assert summary['bin_distribution']['A'] == 1, "Incorrect bin A count"
            assert summary['total_binned'] == 3, "Incorrect total count"
        print(f"✓ Stage 2 results saved to {stage2_dir}")

        # Test Stage 3 save
        print("\n--- Testing Stage 3 (Synthesis) save ---")
        test_qa_dataset = [
            {
                'image': '/path/to/image1.jpg',
                'bin': 'A',
                'question': 'What is shown?',
                'answer': 'A chart',
                'reasoning': 'The image shows a chart with data'
            },
            {
                'image': '/path/to/image2.jpg',
                'bin': 'B',
                'question': 'Where is the car?',
                'answer': 'Left side',
                'reasoning': 'The car is on the left side of the image'
            }
        ]
        pipeline.save_stage3_results(test_qa_dataset)

        stage3_dir = Path(tmpdir) / "intermediate" / "stage3_synthesis"
        assert stage3_dir.exists(), "Stage 3 directory not created"
        assert (stage3_dir / "generated_qa_pairs.jsonl").exists(), "generated_qa_pairs.jsonl not created"
        assert (stage3_dir / "bin_A_qa_pairs.jsonl").exists(), "bin_A_qa_pairs.jsonl not created"
        assert (stage3_dir / "bin_B_qa_pairs.jsonl").exists(), "bin_B_qa_pairs.jsonl not created"
        assert (stage3_dir / "summary.json").exists(), "summary.json not created"

        # Verify content
        with open(stage3_dir / "summary.json") as f:
            summary = json.load(f)
            assert summary['total_generated'] == 2, "Incorrect total count"
            assert summary['by_bin']['A'] == 1, "Incorrect bin A count"
        print(f"✓ Stage 3 results saved to {stage3_dir}")

        # Test Stage 4 save
        print("\n--- Testing Stage 4 (Validation) save ---")
        test_validated_dataset = [
            {
                'image': '/path/to/image1.jpg',
                'bin': 'A',
                'question': 'What is shown?',
                'answer': 'A chart',
                'reasoning': 'The image shows a chart with data'
            }
        ]
        pipeline.save_stage4_results(test_validated_dataset, original_count=2)

        stage4_dir = Path(tmpdir) / "intermediate" / "stage4_validation"
        assert stage4_dir.exists(), "Stage 4 directory not created"
        assert (stage4_dir / "validated_qa_pairs.jsonl").exists(), "validated_qa_pairs.jsonl not created"
        assert (stage4_dir / "summary.json").exists(), "summary.json not created"

        # Verify content
        with open(stage4_dir / "summary.json") as f:
            summary = json.load(f)
            assert summary['original_count'] == 2, "Incorrect original count"
            assert summary['validated_count'] == 1, "Incorrect validated count"
            assert summary['removed_count'] == 1, "Incorrect removed count"
            assert summary['removal_rate_percent'] == 50.0, "Incorrect removal rate"
        print(f"✓ Stage 4 results saved to {stage4_dir}")

        print("\n" + "="*60)
        print("✓ All intermediate save tests passed!")
        print("="*60)

        # Test disabling intermediate saves
        print("\n--- Testing disabled intermediate saves ---")
        config['output']['save_intermediate'] = False
        config_path2 = Path(tmpdir) / "test_config2.yaml"
        with open(config_path2, 'w') as f:
            yaml.dump(config, f)

        pipeline2 = DataSynthesisPipeline(
            config_path=str(config_path2),
            output_dir=tmpdir + "_test2",
            device='cpu'
        )

        print(f"✓ Pipeline initialized with save_intermediate={pipeline2.save_intermediate}")

        # These should not create files
        pipeline2.save_stage1_results(test_filtered_images)
        test2_dir = Path(tmpdir + "_test2") / "intermediate"
        assert not test2_dir.exists(), "Intermediate directory should not be created when disabled"
        print("✓ Intermediate saves correctly disabled")

        print("\n" + "="*60)
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("="*60)


if __name__ == "__main__":
    test_intermediate_saves()
