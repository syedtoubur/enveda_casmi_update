# Attack: isomer-mut-ice (I1b.0)
1. A1 truth leak (blocking): truths loaded only after the top-150 file is frozen; a test checks generation never reads truths.
2. A2 formula drift (blocking): any child formula ≠ parent formula → void.
3. A3 cap still binding (flag): report how many molecules hit 600; reach_all stays a lower bound if many do.
4. A4 frag cut loses the truth before ICE (flag): report the truth's frag rank distribution among all children.
5. A5 runtime (flag): CPU s/molecule at cap 600; projected hidden-test runtime for 400 molecules.
