Mutatest diagnostic summary
===========================
 - Source location: /Users/darianrosebrook/Desktop/Projects/distill/training/losses.py
 - Test commands: ['pytest', 'tests/training/test_losses.py', '-x']
 - Mode: s
 - Excluded files: []
 - N locations input: 20
 - Random seed: None

Random sample details
---------------------
 - Total locations mutated: 20
 - Total locations identified: 282
 - Location sample coverage: 7.09 %


Running time details
--------------------
 - Clean trial 1 run time: 0:00:11.459158
 - Clean trial 2 run time: 0:00:04.045747
 - Mutation trials total run time: 0:02:14.174524

Overall mutation trial summary
==============================
 - SURVIVED: 14
 - DETECTED: 12
 - TOTAL RUNS: 26
 - RUN DATETIME: 2025-11-14 15:11:19.635105


Mutations by result status
==========================


SURVIVED
--------
 - training/losses.py: (l: 53, c: 30) - mutation from <class 'ast.Div'> to <class 'ast.Add'>
 - training/losses.py: (l: 605, c: 8) - mutation from If_Statement to If_False
 - training/losses.py: (l: 639, c: 34) - mutation from <class 'ast.Mult'> to <class 'ast.Add'>
 - training/losses.py: (l: 688, c: 4) - mutation from If_Statement to If_False
 - training/losses.py: (l: 721, c: 46) - mutation from None to False
 - training/losses.py: (l: 767, c: 12) - mutation from If_Statement to If_False
 - training/losses.py: (l: 769, c: 16) - mutation from If_Statement to If_False
 - training/losses.py: (l: 795, c: 8) - mutation from If_Statement to If_True
 - training/losses.py: (l: 851, c: 11) - mutation from <class 'ast.Gt'> to <class 'ast.LtE'>
 - training/losses.py: (l: 966, c: 24) - mutation from <class 'ast.Mult'> to <class 'ast.Sub'>
 - training/losses.py: (l: 967, c: 19) - mutation from <class 'ast.Sub'> to <class 'ast.Mult'>
 - training/losses.py: (l: 1010, c: 20) - mutation from <class 'ast.Sub'> to <class 'ast.Mod'>
 - training/losses.py: (l: 1050, c: 7) - mutation from <class 'ast.Lt'> to <class 'ast.NotEq'>
 - training/losses.py: (l: 1150, c: 11) - mutation from False to None


DETECTED
--------
 - training/losses.py: (l: 53, c: 30) - mutation from <class 'ast.Div'> to <class 'ast.Pow'>
 - training/losses.py: (l: 315, c: 37) - mutation from None to True
 - training/losses.py: (l: 315, c: 37) - mutation from None to False
 - training/losses.py: (l: 605, c: 8) - mutation from If_Statement to If_True
 - training/losses.py: (l: 634, c: 7) - mutation from <class 'ast.And'> to <class 'ast.Or'>
 - training/losses.py: (l: 649, c: 4) - mutation from If_Statement to If_True
 - training/losses.py: (l: 649, c: 4) - mutation from If_Statement to If_False
 - training/losses.py: (l: 649, c: 7) - mutation from <class 'ast.And'> to <class 'ast.Or'>
 - training/losses.py: (l: 680, c: 35) - mutation from <class 'ast.IsNot'> to <class 'ast.Is'>
 - training/losses.py: (l: 688, c: 4) - mutation from If_Statement to If_True
 - training/losses.py: (l: 721, c: 46) - mutation from None to True
 - training/losses.py: (l: 1100, c: 7) - mutation from <class 'ast.And'> to <class 'ast.Or'>