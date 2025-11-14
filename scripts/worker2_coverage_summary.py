#!/usr/bin/env python3
"""
Generate Worker 2 coverage summary report.
"""
import json
from pathlib import Path

def main():
    coverage_file = Path("coverage_output/worker2_coverage.json")
    if not coverage_file.exists():
        print(f"Coverage file not found: {coverage_file}")
        return
    
    with open(coverage_file) as f:
        data = json.load(f)
    
    print("=" * 80)
    print("WORKER 2 COVERAGE SUMMARY - Evaluation Module")
    print("=" * 80)
    print(f"\nTotal Statements: {data['totals']['num_statements']}")
    print(f"Missing Statements: {data['totals']['missing_lines']}")
    print(f"Coverage: {data['totals']['percent_covered']:.2f}%")
    
    # Filter evaluation modules
    eval_files = {
        k: v for k, v in data['files'].items() 
        if 'evaluation/' in k and not k.endswith('__init__.py')
    }
    
    print(f"\nEvaluation Module Coverage ({len(eval_files)} files):")
    print("-" * 80)
    
    # Sort by coverage percentage
    sorted_files = sorted(
        eval_files.items(),
        key=lambda x: x[1]['summary']['percent_covered'],
        reverse=True
    )
    
    for file_path, file_data in sorted_files:
        summary = file_data['summary']
        percent = summary['percent_covered']
        stmts = summary['num_statements']
        missing = summary['missing_lines']
        covered = stmts - missing
        
        # Extract just the filename
        filename = file_path.split('evaluation/')[-1]
        
        print(f"  {filename:50s} {percent:6.1f}% ({covered:4d}/{stmts:4d} stmts)")
    
    print("\n" + "=" * 80)
    print("Worker 2 Test Results:")
    print(f"  Total Tests: 189")
    print(f"  Passed: 62")
    print(f"  Failed: 127")
    print("=" * 80)

if __name__ == "__main__":
    main()

