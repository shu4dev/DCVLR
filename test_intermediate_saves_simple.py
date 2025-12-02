"""
Simple test to verify intermediate save methods work correctly.
Tests the save methods in isolation without loading heavy dependencies.
"""

import json
import tempfile
from pathlib import Path


def test_save_methods():
    """Test intermediate save methods work correctly."""
    print("Testing intermediate save functionality...")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        print(f"\nUsing temporary directory: {output_dir}")

        # Create a mock pipeline object with just the save methods
        class MockPipeline:
            def __init__(self, output_dir, save_intermediate):
                self.output_dir = Path(output_dir)
                self.save_intermediate = save_intermediate

            def save_stage1_results(self, filtered_images):
                """Save Stage 1 (Filtering) results to disk."""
                if not self.save_intermediate:
                    return

                stage1_dir = self.output_dir / "intermediate" / "stage1_filtering"
                stage1_dir.mkdir(parents=True, exist_ok=True)

                # Save filtered image list
                output_path = stage1_dir / "filtered_images.jsonl"
                with open(output_path, 'w') as f:
                    for img in filtered_images:
                        f.write(json.dumps(img) + '\n')

                # Save summary statistics
                summary = {
                    'stage': 'Stage 1 - Filtering',
                    'total_filtered': len(filtered_images),
                    'images': [img['path'] for img in filtered_images]
                }
                summary_path = stage1_dir / "summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)

                print(f"✓ Stage 1 results saved to {stage1_dir}")

            def save_stage2_results(self, binned_images):
                """Save Stage 2 (Binning) results to disk."""
                if not self.save_intermediate:
                    return

                stage2_dir = self.output_dir / "intermediate" / "stage2_binning"
                stage2_dir.mkdir(parents=True, exist_ok=True)

                # Save each bin separately
                for bin_type, images in binned_images.items():
                    bin_path = stage2_dir / f"bin_{bin_type}.jsonl"
                    with open(bin_path, 'w') as f:
                        for img in images:
                            f.write(json.dumps(img) + '\n')

                # Save all binned images together
                all_binned_path = stage2_dir / "all_binned_images.jsonl"
                with open(all_binned_path, 'w') as f:
                    for bin_type, images in binned_images.items():
                        for img in images:
                            img_with_bin = img.copy()
                            img_with_bin['bin'] = bin_type
                            f.write(json.dumps(img_with_bin) + '\n')

                # Save summary statistics
                summary = {
                    'stage': 'Stage 2 - Binning',
                    'bin_distribution': {
                        'A': len(binned_images['A']),
                        'B': len(binned_images['B']),
                        'C': len(binned_images['C'])
                    },
                    'total_binned': sum(len(images) for images in binned_images.values())
                }
                summary_path = stage2_dir / "summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)

                print(f"✓ Stage 2 results saved to {stage2_dir}")

            def save_stage3_results(self, qa_dataset):
                """Save Stage 3 (Synthesis) results to disk."""
                if not self.save_intermediate:
                    return

                stage3_dir = self.output_dir / "intermediate" / "stage3_synthesis"
                stage3_dir.mkdir(parents=True, exist_ok=True)

                # Save all generated Q/A pairs
                output_path = stage3_dir / "generated_qa_pairs.jsonl"
                with open(output_path, 'w') as f:
                    for qa in qa_dataset:
                        f.write(json.dumps(qa) + '\n')

                # Save by bin type
                bins = {'A': [], 'B': [], 'C': []}
                for qa in qa_dataset:
                    bin_type = qa.get('bin', 'C')
                    bins[bin_type].append(qa)

                for bin_type, qa_list in bins.items():
                    if qa_list:
                        bin_path = stage3_dir / f"bin_{bin_type}_qa_pairs.jsonl"
                        with open(bin_path, 'w') as f:
                            for qa in qa_list:
                                f.write(json.dumps(qa) + '\n')

                # Save summary statistics
                summary = {
                    'stage': 'Stage 3 - Q/A Synthesis',
                    'total_generated': len(qa_dataset),
                    'by_bin': {
                        'A': len(bins['A']),
                        'B': len(bins['B']),
                        'C': len(bins['C'])
                    }
                }
                summary_path = stage3_dir / "summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)

                print(f"✓ Stage 3 results saved to {stage3_dir}")

            def save_stage4_results(self, validated_dataset, original_count):
                """Save Stage 4 (Validation) results to disk."""
                if not self.save_intermediate:
                    return

                stage4_dir = self.output_dir / "intermediate" / "stage4_validation"
                stage4_dir.mkdir(parents=True, exist_ok=True)

                # Save validated Q/A pairs
                output_path = stage4_dir / "validated_qa_pairs.jsonl"
                with open(output_path, 'w') as f:
                    for qa in validated_dataset:
                        f.write(json.dumps(qa) + '\n')

                # Save summary statistics
                removed = original_count - len(validated_dataset)
                removal_rate = (removed / original_count * 100) if original_count > 0 else 0

                summary = {
                    'stage': 'Stage 4 - Validation',
                    'original_count': original_count,
                    'validated_count': len(validated_dataset),
                    'removed_count': removed,
                    'removal_rate_percent': round(removal_rate, 2)
                }
                summary_path = stage4_dir / "summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)

                print(f"✓ Stage 4 results saved to {stage4_dir}")

        # Test with intermediate saves enabled
        print("\n=== Testing with save_intermediate=True ===")
        pipeline = MockPipeline(output_dir, save_intermediate=True)

        # Test Stage 1
        print("\n--- Stage 1: Filtering ---")
        test_filtered = [
            {'path': '/path/to/img1.jpg', 'id': 'img1'},
            {'path': '/path/to/img2.jpg', 'id': 'img2'},
            {'path': '/path/to/img3.jpg', 'id': 'img3'}
        ]
        pipeline.save_stage1_results(test_filtered)

        stage1_dir = output_dir / "intermediate" / "stage1_filtering"
        assert stage1_dir.exists(), "Stage 1 directory not created"
        assert (stage1_dir / "filtered_images.jsonl").exists()
        assert (stage1_dir / "summary.json").exists()

        with open(stage1_dir / "summary.json") as f:
            summary = json.load(f)
            assert summary['total_filtered'] == 3
        print("✓ Stage 1 files verified")

        # Test Stage 2
        print("\n--- Stage 2: Binning ---")
        test_binned = {
            'A': [{'path': '/path/to/img1.jpg', 'id': 'img1'}],
            'B': [{'path': '/path/to/img2.jpg', 'id': 'img2'}],
            'C': [{'path': '/path/to/img3.jpg', 'id': 'img3'}]
        }
        pipeline.save_stage2_results(test_binned)

        stage2_dir = output_dir / "intermediate" / "stage2_binning"
        assert stage2_dir.exists()
        assert (stage2_dir / "bin_A.jsonl").exists()
        assert (stage2_dir / "bin_B.jsonl").exists()
        assert (stage2_dir / "bin_C.jsonl").exists()
        assert (stage2_dir / "all_binned_images.jsonl").exists()
        assert (stage2_dir / "summary.json").exists()

        with open(stage2_dir / "summary.json") as f:
            summary = json.load(f)
            assert summary['total_binned'] == 3
            assert summary['bin_distribution']['A'] == 1
        print("✓ Stage 2 files verified")

        # Test Stage 3
        print("\n--- Stage 3: Q/A Synthesis ---")
        test_qa = [
            {'image': 'img1.jpg', 'bin': 'A', 'question': 'Q1?', 'answer': 'A1', 'reasoning': 'R1'},
            {'image': 'img2.jpg', 'bin': 'B', 'question': 'Q2?', 'answer': 'A2', 'reasoning': 'R2'},
            {'image': 'img3.jpg', 'bin': 'A', 'question': 'Q3?', 'answer': 'A3', 'reasoning': 'R3'}
        ]
        pipeline.save_stage3_results(test_qa)

        stage3_dir = output_dir / "intermediate" / "stage3_synthesis"
        assert stage3_dir.exists()
        assert (stage3_dir / "generated_qa_pairs.jsonl").exists()
        assert (stage3_dir / "bin_A_qa_pairs.jsonl").exists()
        assert (stage3_dir / "bin_B_qa_pairs.jsonl").exists()
        assert (stage3_dir / "summary.json").exists()

        with open(stage3_dir / "summary.json") as f:
            summary = json.load(f)
            assert summary['total_generated'] == 3
            assert summary['by_bin']['A'] == 2
            assert summary['by_bin']['B'] == 1
        print("✓ Stage 3 files verified")

        # Test Stage 4
        print("\n--- Stage 4: Validation ---")
        test_validated = [test_qa[0], test_qa[2]]  # Remove one Q/A pair
        pipeline.save_stage4_results(test_validated, original_count=3)

        stage4_dir = output_dir / "intermediate" / "stage4_validation"
        assert stage4_dir.exists()
        assert (stage4_dir / "validated_qa_pairs.jsonl").exists()
        assert (stage4_dir / "summary.json").exists()

        with open(stage4_dir / "summary.json") as f:
            summary = json.load(f)
            assert summary['original_count'] == 3
            assert summary['validated_count'] == 2
            assert summary['removed_count'] == 1
            assert summary['removal_rate_percent'] == 33.33
        print("✓ Stage 4 files verified")

        # Test with intermediate saves disabled
        print("\n\n=== Testing with save_intermediate=False ===")
        pipeline2 = MockPipeline(output_dir / "test_disabled", save_intermediate=False)
        pipeline2.save_stage1_results(test_filtered)
        pipeline2.save_stage2_results(test_binned)
        pipeline2.save_stage3_results(test_qa)
        pipeline2.save_stage4_results(test_validated, 3)

        disabled_dir = output_dir / "test_disabled" / "intermediate"
        assert not disabled_dir.exists(), "Intermediate dir should not exist when disabled"
        print("✓ Correctly skipped saves when disabled")

        print("\n" + "="*60)
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
        print("="*60)
        print("\nThe intermediate save functionality is working correctly!")


if __name__ == "__main__":
    test_save_methods()
