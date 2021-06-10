# Neighbor-joining-algorithm
Implementation of the neighbor joining algorithm.
## Usage
```
python nj.py
```
## Example
Input: distance matrix in PHYLIP format, lower triangle matrix.
```
6
U68589
U68590 0.3371
U68591 0.3609 0.3782
U68592 0.4155 0.3197 0.4148
U68593 0.2872 0.1690 0.3361 0.2842
U68594 0.2970 0.3293 0.3563 0.3325 0.2768
```
Output: tree (with branch lengths) in Newick format
```\(\(U68591:0.2008125,U68589:0.16008749999999997):0.005718749999999988,\(\(\(U68593:0.062,U68590:0.10700000000000004\):0.03473333333333334,U68592:0.1827166666666667\):0.0320375,U68594:0.13476250000000006\):0.005718749999999988\);
```
