# Domain Scenarios

This file is generated from `scripts/domain/{tomato,wastesorting}/scene_XX.yaml`.

<!-- SCENARIO_SUMMARY_START -->

## Tomato Harvest

| Scene | Tomatoes | Stems | Distribution | Labels | Placement |
|---:|---:|---:|---|---|---|
| 01 | 4 | 2 | 2/2 | ripe=2, rotten=1, unripe=1 | stem_01(t1:ripe, t2:rotten); stem_02(t3:ripe, t4:unripe) |
| 02 | 4 | 2 | 2/2 | ripe=2, rotten=1, unripe=1 | stem_01(t1:ripe, t2:unripe); stem_02(t3:rotten, t4:ripe) |
| 03 | 4 | 2 | 2/2 | ripe=2, rotten=1, unripe=1 | stem_01(t1:ripe, t2:unripe); stem_02(t3:ripe, t4:rotten) |
| 04 | 4 | 2 | 2/2 | ripe=2, rotten=1, unripe=1 | stem_01(t1:rotten, t2:ripe); stem_02(t3:ripe, t4:unripe) |
| 05 | 4 | 2 | 2/2 | ripe=2, rotten=1, unripe=1 | stem_01(t1:rotten, t2:unripe); stem_02(t3:ripe, t4:ripe) |
| 06 | 6 | 3 | 2/2/2 | ripe=2, rotten=2, unripe=2 | stem_01(t1:rotten, t2:unripe); stem_02(t3:unripe, t4:ripe); stem_03(t5:rotten, t6:ripe) |
| 07 | 6 | 3 | 1/2/3 | ripe=2, rotten=2, unripe=2 | stem_01(t1:rotten); stem_02(t2:ripe, t3:rotten); stem_03(t4:ripe, t5:unripe, t6:unripe) |
| 08 | 6 | 3 | 2/1/3 | ripe=2, rotten=2, unripe=2 | stem_01(t1:rotten, t2:unripe); stem_02(t3:rotten); stem_03(t4:unripe, t5:ripe, t6:ripe) |
| 09 | 6 | 3 | 3/1/2 | ripe=2, rotten=2, unripe=2 | stem_01(t1:rotten, t2:ripe, t3:ripe); stem_02(t4:rotten); stem_03(t5:unripe, t6:unripe) |
| 10 | 6 | 3 | 2/2/2 | ripe=2, rotten=2, unripe=2 | stem_01(t1:unripe, t2:rotten); stem_02(t3:unripe, t4:rotten); stem_03(t5:ripe, t6:ripe) |
| 11 | 8 | 4 | 2/2/2/2 | ripe=2, rotten=3, unripe=3 | stem_01(t1:unripe, t2:rotten); stem_02(t3:ripe, t4:unripe); stem_03(t5:rotten, t6:rotten); stem_04(t7:ripe, t8:unripe) |
| 12 | 8 | 4 | 1/2/3/2 | ripe=3, rotten=3, unripe=2 | stem_01(t1:rotten); stem_02(t2:ripe, t3:ripe); stem_03(t4:rotten, t5:unripe, t6:rotten); stem_04(t7:unripe, t8:ripe) |
| 13 | 8 | 4 | 2/1/2/3 | ripe=3, rotten=3, unripe=2 | stem_01(t1:rotten, t2:rotten); stem_02(t3:rotten); stem_03(t4:ripe, t5:ripe); stem_04(t6:ripe, t7:unripe, t8:unripe) |
| 14 | 8 | 4 | 3/2/1/2 | ripe=3, rotten=2, unripe=3 | stem_01(t1:unripe, t2:unripe, t3:ripe); stem_02(t4:rotten, t5:ripe); stem_03(t6:ripe); stem_04(t7:unripe, t8:rotten) |
| 15 | 8 | 4 | 2/2/2/2 | ripe=2, rotten=3, unripe=3 | stem_01(t1:unripe, t2:rotten); stem_02(t3:unripe, t4:ripe); stem_03(t5:rotten, t6:rotten); stem_04(t7:ripe, t8:unripe) |
| 16 | 8 | 3 | 3/3/2 | ripe=3, rotten=3, unripe=2 | stem_01(t1:rotten, t2:rotten, t3:ripe); stem_02(t4:unripe, t5:ripe, t6:rotten); stem_03(t7:unripe, t8:ripe) |
| 17 | 8 | 3 | 3/3/2 | ripe=3, rotten=3, unripe=2 | stem_01(t1:rotten, t2:rotten, t3:unripe); stem_02(t4:ripe, t5:unripe, t6:ripe); stem_03(t7:ripe, t8:rotten) |
| 18 | 8 | 3 | 2/4/2 | ripe=3, rotten=3, unripe=2 | stem_01(t1:ripe, t2:unripe); stem_02(t3:ripe, t4:rotten, t5:unripe, t6:rotten); stem_03(t7:ripe, t8:rotten) |
| 19 | 8 | 3 | 2/2/4 | ripe=3, rotten=3, unripe=2 | stem_01(t1:unripe, t2:rotten); stem_02(t3:ripe, t4:ripe); stem_03(t5:rotten, t6:ripe, t7:rotten, t8:unripe) |
| 20 | 8 | 3 | 2/3/3 | ripe=3, rotten=2, unripe=3 | stem_01(t1:rotten, t2:ripe); stem_02(t3:ripe, t4:ripe, t5:unripe); stem_03(t6:rotten, t7:unripe, t8:unripe) |
| 21 | 12 | 4 | 3/3/3/3 | ripe=4, rotten=4, unripe=4 | stem_01(t1:rotten, t2:ripe, t3:ripe); stem_02(t4:unripe, t5:ripe, t6:unripe); stem_03(t7:unripe, t8:rotten, t9:rotten); stem_04(t10:unripe, t11:ripe, t12:rotten) |
| 22 | 12 | 4 | 4/3/3/2 | ripe=4, rotten=4, unripe=4 | stem_01(t1:unripe, t2:ripe, t3:unripe, t4:unripe); stem_02(t5:unripe, t6:ripe, t7:rotten); stem_03(t8:rotten, t9:rotten, t10:rotten); stem_04(t11:ripe, t12:ripe) |
| 23 | 12 | 4 | 3/4/3/2 | ripe=4, rotten=4, unripe=4 | stem_01(t1:rotten, t2:ripe, t3:rotten); stem_02(t4:unripe, t5:rotten, t6:rotten, t7:unripe); stem_03(t8:ripe, t9:ripe, t10:ripe); stem_04(t11:unripe, t12:unripe) |
| 24 | 12 | 4 | 4/2/2/4 | ripe=4, rotten=4, unripe=4 | stem_01(t1:unripe, t2:rotten, t3:unripe, t4:rotten); stem_02(t5:ripe, t6:unripe); stem_03(t7:ripe, t8:rotten); stem_04(t9:rotten, t10:unripe, t11:ripe, t12:ripe) |
| 25 | 12 | 4 | 3/3/3/3 | ripe=4, rotten=4, unripe=4 | stem_01(t1:unripe, t2:ripe, t3:ripe); stem_02(t4:rotten, t5:unripe, t6:rotten); stem_03(t7:rotten, t8:unripe, t9:ripe); stem_04(t10:unripe, t11:rotten, t12:ripe) |

## Waste Sorting

| Scene | Wastes | Labels | Occlusion |
|---:|---:|---|---|
| 01 | 4 | general=1, paper=1, plastic=1, can=1 | - |
| 02 | 4 | general=1, paper=1, plastic=0, can=2 | on(3,4) |
| 03 | 4 | general=1, paper=1, plastic=2, can=0 | on(3,4) |
| 04 | 4 | general=1, paper=2, plastic=0, can=1 | on(1,3), on(2,4) |
| 05 | 4 | general=0, paper=2, plastic=0, can=2 | on(1,3), on(2,4) |
| 06 | 6 | general=1, paper=1, plastic=2, can=2 | on(1,4), on(2,5), on(3,6) |
| 07 | 6 | general=1, paper=2, plastic=2, can=1 | on(2,5), on(3,6) |
| 08 | 6 | general=1, paper=2, plastic=1, can=2 | on(1,3), on(2,4), on(3,5), on(4,6) |
| 09 | 6 | general=2, paper=1, plastic=1, can=2 | on(2,4), on(3,5), on(5,6) |
| 10 | 6 | general=1, paper=2, plastic=1, can=2 | on(1,4), on(2,5), on(3,6) |
| 11 | 8 | general=2, paper=2, plastic=2, can=2 | on(1,5), on(2,6), on(3,7), on(4,8) |
| 12 | 8 | general=2, paper=2, plastic=2, can=2 | on(1,4), on(2,5), on(3,6), on(5,7), on(6,8) |
| 13 | 8 | general=2, paper=2, plastic=2, can=2 | on(1,5), on(2,6), on(4,7), on(7,8) |
| 14 | 8 | general=2, paper=2, plastic=2, can=2 | on(2,6), on(3,7), on(4,8) |
| 15 | 8 | general=2, paper=2, plastic=2, can=2 | on(1,5), on(2,6), on(3,7), on(4,8) |
| 16 | 10 | general=3, paper=2, plastic=2, can=3 | on(1,6), on(2,7), on(3,8), on(4,9), on(5,10) |
| 17 | 10 | general=3, paper=2, plastic=3, can=2 | on(1,6), on(2,7), on(3,8), on(4,9), on(8,10) |
| 18 | 10 | general=2, paper=2, plastic=3, can=3 | on(2,7), on(3,8), on(7,9), on(8,10) |
| 19 | 10 | general=3, paper=3, plastic=2, can=2 | on(2,5), on(3,6), on(4,7), on(5,8), on(6,9), on(7,10) |
| 20 | 10 | general=2, paper=2, plastic=3, can=3 | on(1,5), on(2,6), on(3,7), on(4,8), on(5,9), on(8,10) |
| 21 | 20 | general=5, paper=5, plastic=5, can=5 | on(1,6), on(2,7), on(3,8), on(4,9), on(5,10), on(6,11), on(7,12), on(8,13), on(9,14), on(10,15), on(11,16), on(12,17), on(13,18), on(14,19), on(15,20) |
| 22 | 20 | general=5, paper=5, plastic=5, can=5 | on(1,7), on(2,8), on(3,13), on(4,9), on(5,10), on(7,11), on(8,12), on(9,14), on(10,15), on(11,16), on(12,17), on(13,18), on(14,19), on(15,20) |
| 23 | 20 | general=5, paper=5, plastic=5, can=5 | on(2,6), on(3,7), on(4,11), on(5,8), on(6,9), on(7,10), on(8,12), on(9,13), on(10,14), on(11,15), on(12,16), on(13,17), on(14,18), on(15,19), on(16,20) |
| 24 | 20 | general=5, paper=5, plastic=5, can=5 | on(1,7), on(2,8), on(3,9), on(4,10), on(5,11), on(8,12), on(9,15), on(10,13), on(11,20), on(12,14), on(13,16), on(14,17), on(15,18), on(16,19) |
| 25 | 20 | general=5, paper=5, plastic=5, can=5 | on(1,9), on(2,7), on(3,11), on(4,12), on(5,13), on(6,8), on(7,10), on(8,14), on(9,15), on(10,16), on(11,17), on(12,18), on(13,19), on(14,20) |

<!-- SCENARIO_SUMMARY_END -->
