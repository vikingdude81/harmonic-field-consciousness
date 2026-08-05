#!/usr/bin/env bash
# board.sh - maintain the BEARERS THEORY GitHub project (vikingdude81/projects/1)
#
# Usage:
#   ./board.sh ls
#   ./board.sh add "<title>" <Type> <Result> <Prediction> <Status> "<detail>"
#   ./board.sh update "<title-substring>" <field> <value>
#
#   fields for update: type | result | prediction | status | detail
#   Type   : Paper Experiment Framework Lever Doc/Infra
#   Result : Confirmed Null Partial Open N/A
#   Pred   : P1 P2 P3 P4 P5 P7 PSD None
#   Status : Todo "In Progress" Done
#
# Examples:
#   ./board.sh update "Calibrate viability" status "In Progress"
#   ./board.sh update "Calibrate viability" result Partial
#   ./board.sh add "New probe X" Experiment Open P2 Todo cca/RESULTS_X.md
set -euo pipefail

O=vikingdude81
NUM=1
PROJECT=PVT_kwHOBb3Bwc4BfZjO

declare -A FIELD=(
  [type]=PVTSSF_lAHOBb3Bwc4BfZjOzhZt1W0
  [result]=PVTSSF_lAHOBb3Bwc4BfZjOzhZt1Xw
  [prediction]=PVTSSF_lAHOBb3Bwc4BfZjOzhZt1ag
  [status]=PVTSSF_lAHOBb3Bwc4BfZjOzhZst_U
  [detail]=PVTF_lAHOBb3Bwc4BfZjOzhZt1bM
)

# field:value -> single-select option id
declare -A OPT=(
  [type:Paper]=e7ba1490 [type:Experiment]=1af755d5 [type:Framework]=30cc875e [type:Lever]=1f8984e8 [type:Doc/Infra]=b2b6936e
  [result:Confirmed]=8b8382b3 [result:Null]=af73802f [result:Partial]=b0da3269 [result:Open]=a35e2c48 [result:N/A]=9798f52b
  [prediction:P1]=da37ea47 [prediction:P2]=c6c441fd [prediction:P3]=fef6134a [prediction:P4]=e955f577 [prediction:P5]=6cbebe94 [prediction:P7]=1ff2ea3a [prediction:PSD]=f729f46d [prediction:None]=b8e5e5c1
  [status:Todo]=f75ad846 ["status:In Progress"]=47fc9ee4 [status:Done]=98236657
)

resolve() {  # title substring -> item id (first match)
  gh api graphql -f query='query($o:String!,$n:Int!){user(login:$o){projectV2(number:$n){items(first:100){nodes{id content{... on DraftIssue{title}}}}}}}' \
    -f o="$O" -F n="$NUM" --jq ".data.user.projectV2.items.nodes[] | select(.content.title | test(\"$1\"; \"i\")) | .id" | head -1
}

edit_select() { gh project item-edit --project-id "$PROJECT" --id "$1" --field-id "${FIELD[$2]}" --single-select-option-id "$3" >/dev/null; }

cmd=${1:-ls}; shift || true
case "$cmd" in
  ls)
    gh api graphql -f query='query($o:String!,$n:Int!){user(login:$o){projectV2(number:$n){items(first:100){nodes{content{... on DraftIssue{title}} fieldValues(first:10){nodes{... on ProjectV2ItemFieldSingleSelectValue{name field{... on ProjectV2FieldCommon{name}}}}}}}}}}' \
      -f o="$O" -F n="$NUM" --jq '.data.user.projectV2.items.nodes[] | (.content.title) as $t | (.fieldValues.nodes | map(select(.field.name=="Status"))[0].name // "-") as $s | (.fieldValues.nodes | map(select(.field.name=="Result"))[0].name // "-") as $r | "\($s|.[0:11] + "           " | .[0:11])  \($r|.[0:9] + "         " | .[0:9])  \($t)"'
    ;;
  add)
    title=$1; type=$2; result=$3; pred=$4; status=$5; detail=${6:-}
    id=$(gh project item-create "$NUM" --owner "$O" --title "$title" --format json --jq '.id')
    edit_select "$id" type       "${OPT[type:$type]}"
    edit_select "$id" result     "${OPT[result:$result]}"
    edit_select "$id" prediction "${OPT[prediction:$pred]}"
    edit_select "$id" status     "${OPT[status:$status]}"
    [ -n "$detail" ] && gh project item-edit --project-id "$PROJECT" --id "$id" --field-id "${FIELD[detail]}" --text "$detail" >/dev/null
    echo "added: $title"
    ;;
  update)
    match=$1; field=$2; value=$3
    id=$(resolve "$match"); [ -z "$id" ] && { echo "no item matching: $match" >&2; exit 1; }
    if [ "$field" = "detail" ]; then
      gh project item-edit --project-id "$PROJECT" --id "$id" --field-id "${FIELD[detail]}" --text "$value" >/dev/null
    else
      opt=${OPT[$field:$value]:-}; [ -z "$opt" ] && { echo "unknown $field value: $value" >&2; exit 1; }
      edit_select "$id" "$field" "$opt"
    fi
    echo "updated: $match  ($field -> $value)"
    ;;
  *) echo "usage: ls | add | update  (see header)" >&2; exit 1 ;;
esac
