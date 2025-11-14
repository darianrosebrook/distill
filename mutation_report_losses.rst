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
 - Clean trial 1 run time: 0:00:05.699826
 - Clean trial 2 run time: 0:00:04.377442
 - Mutation trials total run time: 0:03:25.176691

Overall mutation trial summary
==============================
 - DETECTED: 35
 - SURVIVED: 13
 - TOTAL RUNS: 48
 - RUN DATETIME: 2025-11-14 15:23:21.553687


Mutations by result status
==========================


SURVIVED
--------
 - training/losses.py: (l: 370, c: 9) - mutation from <class 'ast.Eq'> to <class 'ast.LtE'>
 - training/losses.py: (l: 546, c: 45) - mutation from None to True
 - training/losses.py: (l: 688, c: 4) - mutation from If_Statement to If_False
 - training/losses.py: (l: 765, c: 15) - mutation from <class 'ast.GtE'> to <class 'ast.Gt'>
 - training/losses.py: (l: 818, c: 12) - mutation from If_Statement to If_False
 - training/losses.py: (l: 833, c: 23) - mutation from <class 'ast.And'> to <class 'ast.Or'>
 - training/losses.py: (l: 846, c: 52) - mutation from <class 'ast.Mult'> to <class 'ast.Div'>
 - training/losses.py: (l: 952, c: 51) - mutation from <class 'ast.Add'> to <class 'ast.Sub'>
 - training/losses.py: (l: 1010, c: 20) - mutation from <class 'ast.Sub'> to <class 'ast.Mod'>
 - training/losses.py: (l: 1083, c: 28) - mutation from <class 'ast.In'> to <class 'ast.NotIn'>
 - training/losses.py: (l: 1088, c: 8) - mutation from <class 'ast.In'> to <class 'ast.NotIn'>
 - training/losses.py: (l: 1093, c: 8) - mutation from AugAssign_Add to AugAssign_Div
 - training/losses.py: (l: 1148, c: 19) - mutation from True to None


DETECTED
--------
 - training/losses.py: (l: 308, c: 19) - mutation from <class 'ast.Add'> to <class 'ast.Mult'>
 - training/losses.py: (l: 308, c: 19) - mutation from <class 'ast.Add'> to <class 'ast.FloorDiv'>
 - training/losses.py: (l: 308, c: 19) - mutation from <class 'ast.Add'> to <class 'ast.Sub'>
 - training/losses.py: (l: 308, c: 19) - mutation from <class 'ast.Add'> to <class 'ast.Div'>
 - training/losses.py: (l: 308, c: 19) - mutation from <class 'ast.Add'> to <class 'ast.Pow'>
 - training/losses.py: (l: 308, c: 19) - mutation from <class 'ast.Add'> to <class 'ast.Mod'>
 - training/losses.py: (l: 355, c: 17) - mutation from <class 'ast.Div'> to <class 'ast.FloorDiv'>
 - training/losses.py: (l: 355, c: 17) - mutation from <class 'ast.Div'> to <class 'ast.Mult'>
 - training/losses.py: (l: 355, c: 17) - mutation from <class 'ast.Div'> to <class 'ast.Sub'>
 - training/losses.py: (l: 355, c: 17) - mutation from <class 'ast.Div'> to <class 'ast.Mod'>
 - training/losses.py: (l: 355, c: 17) - mutation from <class 'ast.Div'> to <class 'ast.Pow'>
 - training/losses.py: (l: 355, c: 17) - mutation from <class 'ast.Div'> to <class 'ast.Add'>
 - training/losses.py: (l: 370, c: 9) - mutation from <class 'ast.Eq'> to <class 'ast.Gt'>
 - training/losses.py: (l: 370, c: 9) - mutation from <class 'ast.Eq'> to <class 'ast.NotEq'>
 - training/losses.py: (l: 627, c: 7) - mutation from <class 'ast.And'> to <class 'ast.Or'>
 - training/losses.py: (l: 627, c: 7) - mutation from <class 'ast.IsNot'> to <class 'ast.Is'>
 - training/losses.py: (l: 641, c: 7) - mutation from <class 'ast.And'> to <class 'ast.Or'>
 - training/losses.py: (l: 765, c: 15) - mutation from <class 'ast.GtE'> to <class 'ast.NotEq'>
 - training/losses.py: (l: 765, c: 15) - mutation from <class 'ast.GtE'> to <class 'ast.Eq'>
 - training/losses.py: (l: 765, c: 15) - mutation from <class 'ast.GtE'> to <class 'ast.Lt'>
 - training/losses.py: (l: 765, c: 15) - mutation from <class 'ast.GtE'> to <class 'ast.LtE'>
 - training/losses.py: (l: 846, c: 52) - mutation from <class 'ast.Mult'> to <class 'ast.FloorDiv'>
 - training/losses.py: (l: 846, c: 52) - mutation from <class 'ast.Mult'> to <class 'ast.Add'>
 - training/losses.py: (l: 846, c: 52) - mutation from <class 'ast.Mult'> to <class 'ast.Mod'>
 - training/losses.py: (l: 846, c: 52) - mutation from <class 'ast.Mult'> to <class 'ast.Sub'>
 - training/losses.py: (l: 848, c: 25) - mutation from <class 'ast.Add'> to <class 'ast.Div'>
 - training/losses.py: (l: 848, c: 25) - mutation from <class 'ast.Add'> to <class 'ast.Mod'>
 - training/losses.py: (l: 848, c: 25) - mutation from <class 'ast.Add'> to <class 'ast.Pow'>
 - training/losses.py: (l: 848, c: 25) - mutation from <class 'ast.Add'> to <class 'ast.Mult'>
 - training/losses.py: (l: 848, c: 25) - mutation from <class 'ast.Add'> to <class 'ast.FloorDiv'>
 - training/losses.py: (l: 848, c: 25) - mutation from <class 'ast.Add'> to <class 'ast.Sub'>
 - training/losses.py: (l: 1010, c: 20) - mutation from <class 'ast.Sub'> to <class 'ast.Add'>
 - training/losses.py: (l: 1010, c: 20) - mutation from <class 'ast.Sub'> to <class 'ast.Div'>
 - training/losses.py: (l: 1050, c: 4) - mutation from If_Statement to If_True
 - training/losses.py: (l: 1050, c: 4) - mutation from If_Statement to If_False