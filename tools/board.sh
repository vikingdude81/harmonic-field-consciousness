#!/usr/bin/env bash
# board.sh - maintain the two GitHub project boards for the research program.
#
#   bearer = BEARERS THEORY        (vikingdude81/projects/1)  has a Prediction field
#   lbl    = Life Before Language  (vikingdude81/projects/2)  no Prediction field
#
# Usage:
#   ./board.sh <bearer|lbl> ls
#   ./board.sh <bearer|lbl> add "<title>" <Type> <Result> <Status> "<detail>" [Prediction]
#   ./board.sh <bearer|lbl> update "<title-substring>" <field> <value>
#
#   Type   : Paper Experiment Framework Lever Doc/Infra
#   Result : Confirmed Null Partial Open N/A
#   Status : Todo "In Progress" Done
#   Pred   : P1 P2 P3 P4 P5 P7 PSD None   (bearer board only)
#   update fields: type | result | status | detail | prediction(bearer only)
#
# Examples:
#   ./board.sh bearer update "Calibrate viability" status "In Progress"
#   ./board.sh bearer add "New probe X" Experiment Open Todo cca/RESULTS_X.md P2
#   ./board.sh lbl    update "Venue choice" status "In Progress"
set -euo pipefail

O=vikingdude81
board=${1:-}; shift || true
declare -A FIELD OPT

case "$board" in
  bearer)
    NUM=1; PROJECT=PVT_kwHOBb3Bwc4BfZjO; HAS_PRED=1
    FIELD=(
      [type]=PVTSSF_lAHOBb3Bwc4BfZjOzhZt1W0 [result]=PVTSSF_lAHOBb3Bwc4BfZjOzhZt1Xw
      [prediction]=PVTSSF_lAHOBb3Bwc4BfZjOzhZt1ag [status]=PVTSSF_lAHOBb3Bwc4BfZjOzhZst_U
      [detail]=PVTF_lAHOBb3Bwc4BfZjOzhZt1bM )
    OPT=(
      [type:Paper]=e7ba1490 [type:Experiment]=1af755d5 [type:Framework]=30cc875e [type:Lever]=1f8984e8 [type:Doc/Infra]=b2b6936e
      [result:Confirmed]=8b8382b3 [result:Null]=af73802f [result:Partial]=b0da3269 [result:Open]=a35e2c48 [result:N/A]=9798f52b
      [prediction:P1]=da37ea47 [prediction:P2]=c6c441fd [prediction:P3]=fef6134a [prediction:P4]=e955f577 [prediction:P5]=6cbebe94 [prediction:P7]=1ff2ea3a [prediction:PSD]=f729f46d [prediction:None]=b8e5e5c1
      [status:Todo]=f75ad846 ["status:In Progress"]=47fc9ee4 [status:Done]=98236657 )
    ;;
  lbl)
    NUM=2; PROJECT=PVT_kwHOBb3Bwc4BfZje; HAS_PRED=0
    FIELD=(
      [type]=PVTSSF_lAHOBb3Bwc4BfZjezhZt9es [result]=PVTSSF_lAHOBb3Bwc4BfZjezhZt9ew
      [status]=PVTSSF_lAHOBb3Bwc4BfZjezhZsuNs [detail]=PVTF_lAHOBb3Bwc4BfZjezhZt9e0 )
    OPT=(
      [type:Paper]=6e4b2019 [type:Experiment]=aa52dc9b [type:Framework]=97b50325 [type:Lever]=d051082d [type:Doc/Infra]=0810f584
      [result:Confirmed]=273a7cbe [result:Null]=89a1b10e [result:Partial]=b924f24a [result:Open]=dd8901ed [result:N/A]=14b765c3
      [status:Todo]=f75ad846 ["status:In Progress"]=47fc9ee4 [status:Done]=98236657 )
    ;;
  *) echo "usage: board.sh <bearer|lbl> <ls|add|update> ...  (see header)" >&2; exit 1 ;;
esac

resolve() {  # title substring -> item id (first match)
  gh api graphql -f query='query($o:String!,$n:Int!){user(login:$o){projectV2(number:$n){items(first:100){nodes{id content{... on DraftIssue{title}}}}}}}' \
    -f o="$O" -F n="$NUM" --jq ".data.user.projectV2.items.nodes[] | select(.content.title | test(\"$1\"; \"i\")) | .id" | head -1
}
sel() { gh project item-edit --project-id "$PROJECT" --id "$1" --field-id "${FIELD[$2]}" --single-select-option-id "$3" >/dev/null; }
txt() { gh project item-edit --project-id "$PROJECT" --id "$1" --field-id "${FIELD[detail]}" --text "$2" >/dev/null; }

cmd=${1:-ls}; shift || true
case "$cmd" in
  ls)
    gh api graphql -f query='query($o:String!,$n:Int!){user(login:$o){projectV2(number:$n){items(first:100){nodes{content{... on DraftIssue{title}} fieldValues(first:10){nodes{... on ProjectV2ItemFieldSingleSelectValue{name field{... on ProjectV2FieldCommon{name}}}}}}}}}}' \
      -f o="$O" -F n="$NUM" --jq '.data.user.projectV2.items.nodes[] | (.content.title) as $t | (.fieldValues.nodes | map(select(.field.name=="Status"))[0].name // "-") as $s | (.fieldValues.nodes | map(select(.field.name=="Result"))[0].name // "-") as $r | "\(($s + "           ")[0:11])  \(($r + "         ")[0:9])  \($t)"'
    ;;
  add)
    title=$1; type=$2; result=$3; status=$4; detail=${5:-}; pred=${6:-}
    id=$(gh project item-create "$NUM" --owner "$O" --title "$title" --format json --jq '.id')
    sel "$id" type "${OPT[type:$type]}"; sel "$id" result "${OPT[result:$result]}"; sel "$id" status "${OPT[status:$status]}"
    [ -n "$detail" ] && txt "$id" "$detail"
    if [ -n "$pred" ] && [ "$HAS_PRED" = 1 ]; then sel "$id" prediction "${OPT[prediction:$pred]}"; fi
    echo "added ($board): $title"
    ;;
  update)
    match=$1; field=$2; value=$3
    [ "$field" = prediction ] && [ "$HAS_PRED" != 1 ] && { echo "no prediction field on '$board'" >&2; exit 1; }
    id=$(resolve "$match"); [ -z "$id" ] && { echo "no item matching: $match" >&2; exit 1; }
    if [ "$field" = detail ]; then txt "$id" "$value"
    else opt=${OPT[$field:$value]:-}; [ -z "$opt" ] && { echo "unknown $field value: $value" >&2; exit 1; }; sel "$id" "$field" "$opt"; fi
    echo "updated ($board): $match  ($field -> $value)"
    ;;
  *) echo "usage: board.sh <bearer|lbl> <ls|add|update> ...  (see header)" >&2; exit 1 ;;
esac
