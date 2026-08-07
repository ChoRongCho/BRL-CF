# simple run for baseline model "Attr-POMDP"

python3 scripts/baseline/attr_pomdp/main.py \
  --domain tomato \
  --scene 01 \
    --feedback-source oracle \
  --query-cost 1.0 \
  --max-depth 2 \
  --max-step 25

# python3 scripts/baseline/attr_pomdp/main.py \
#   --domain wastesorting \
#   --scene 01 \
#     --feedback-source oracle \
#   --query-cost 1.0 \
#   --max-depth 2 \
#   --max-step 25
