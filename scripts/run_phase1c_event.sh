#!/usr/bin/env bash
# Phase 1C: run ONE frozen pilot event through the same five experimental steps.
# Usage: bash scripts/run_phase1c_event.sh E001 [--end-frame 202] [--dry-run] [--yes] [--force]
set -Eeuo pipefail

fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
info() { printf '\n\033[1;36m%s\033[0m\n' "$*"; }
trap 'code=$?; printf "\nStopped at line %s (exit status %s). Existing results were kept.\n" "$LINENO" "$code" >&2' ERR

[[ $# -ge 1 ]] || fail "Usage: bash scripts/run_phase1c_event.sh E001 [--end-frame NUMBER] [--dry-run] [--yes] [--force]"
EVENT="$1"; shift
[[ "$EVENT" =~ ^E0(0[1-9]|1[0-2])$ ]] || fail "Use exactly one frozen event ID, E001 through E012."
END_OVERRIDE=""
DRY_RUN=0
AUTO_CONFIRM=0
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --end-frame)
      [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || fail "--end-frame needs a positive integer."
      END_OVERRIDE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes) AUTO_CONFIRM=1; shift ;;
    --force) FORCE=1; shift ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SAM_PY="${SAM_PY:-$HOME/miniconda3/envs/vehicle_sam21/bin/python}"
DAM_PY="${DAM_PY:-$HOME/miniconda3/envs/vehicle_dam4sam/bin/python}"
[[ -x "$SAM_PY" ]] || fail "SAM2.1 environment Python missing: $SAM_PY"
[[ -x "$DAM_PY" ]] || fail "DAM4SAM environment Python missing: $DAM_PY"
for path in src/evaluation/event_loader.py src/evaluation/event_evaluator.py src/run_phase1_event.py data/processed/pilot_events/pilot_event_log.csv; do
  [[ -f "$path" ]] || fail "Missing project file: $path"
done
EVENT_DIR="$ROOT/results/phase1/$EVENT"

info "STEP 1/5 — Validate frozen $EVENT and preview the initial target"
# Python prints just the chosen end-frame to stdout; descriptive output goes to stderr.
END_FRAME="$("$SAM_PY" - "$EVENT" "$END_OVERRIDE" "$ROOT" <<'PY'
import sys, json
from pathlib import Path
from PIL import Image, ImageDraw
root = Path(sys.argv[3]); sys.path.insert(0, str(root / 'src'))
from evaluation.event_loader import load_event

event_id, end_override = sys.argv[1], sys.argv[2]
e = load_event(event_id)  # Existing loader is authoritative; never modify frozen CSV.
frame_dir = Path(e['image_dir'])
frames = {int(''.join(c for c in p.stem if c.isdigit())): p for p in frame_dir.iterdir()
          if p.suffix.lower() in ('.jpg', '.jpeg', '.png') and any(c.isdigit() for c in p.stem)}
if not frames: raise SystemExit('No images found for this sequence')
first, last = min(frames), max(frames)
selection = int(e['selection_frame'])
reappearance, fully = e['reappearance_frame'], e['fully_visible_frame']
# Preserve reappearance+30, but do not truncate a later recorded fully-visible frame.
# For no recorded reappearance/visibility, use the sequence's last image.
landmarks = ([int(reappearance) + 30] if reappearance is not None else [])
if fully is not None: landmarks.append(int(fully))
end = int(end_override) if end_override else min(last, max(landmarks)) if landmarks else last
if not (first <= selection < end <= last):
    raise SystemExit(f'Invalid frame interval: selection={selection}, end={end}, images={first}..{last}')
missing = [f for f in range(selection, end+1) if f not in frames]
if missing: raise SystemExit(f'{len(missing)} missing images in requested interval; examples: {missing[:5]}')
xyxy = e['selection_box']['xyxy']
if len(xyxy) != 4 or xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
    raise SystemExit('Invalid selection box')
base = root / 'results' / 'phase1' / event_id
base.mkdir(parents=True, exist_ok=True)
preview = base / 'selection_preview.jpg'
image = Image.open(frames[selection]).convert('RGB')
draw = ImageDraw.Draw(image)
draw.rectangle(tuple(xyxy), outline='#00ff00', width=3)
draw.text((max(0, xyxy[0]), max(0, xyxy[1]-16)),
          f'{event_id} target {e["target_id"]} / frame {selection}', fill='#00ff00', stroke_width=1, stroke_fill='black')
image.save(preview, quality=92)
manifest = {
    'event_id': event_id, 'sequence': e['sequence'], 'target_id': e['target_id'],
    'scenario': e['scenario'], 'verification_status': e['verification_status'],
    'selection_frame': selection, 'selection_box_xyxy': xyxy,
    'occlusion_start': e['occlusion_start'],
    'maximum_occlusion_start': e['maximum_occlusion_start'],
    'reappearance_frame': reappearance, 'fully_visible_frame': fully,
    'run_end_frame': end, 'future_frames': end-selection,
    'end_frame_rule': 'explicit --end-frame' if end_override else 'reappearance+30 / fully-visible / sequence-end',
    'selection_preview': str(preview.relative_to(root)),
    'visual_verification_required': True,
}
(base / 'experiment_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(f'Event: {event_id} | {e["scenario"]} | {e["verification_status"]}', file=sys.stderr)
print(f'Sequence: {e["sequence"]} | target ID: {e["target_id"]}', file=sys.stderr)
print(f'Initialize: frame {selection}, bbox xyxy={[round(v,1) for v in xyxy]}', file=sys.stderr)
print(f'Occlusion: {e["occlusion_start"]} | maximum: {e["maximum_occlusion_start"]} | reappearance: {reappearance}', file=sys.stderr)
print(f'BOTH trackers will process frames {selection+1}..{end} ({end-selection} future frames).', file=sys.stderr)
print(f'PREVIEW: {preview}', file=sys.stderr)
if e['verification_status'] != 'manually_reviewed':
    print('WARNING: Event verification is provisional; visually verify all relevant labels.', file=sys.stderr)
print(end)
PY
)" || fail "Preflight failed. Check event_loader.py, annotations and selection frame."
[[ "$END_FRAME" =~ ^[0-9]+$ ]] || fail "Could not resolve common end frame."

if (( DRY_RUN )); then
  info "DRY RUN COMPLETE — no trackers/evaluator ran; check the preview and manifest."
  exit 0
fi

if (( ! AUTO_CONFIRM )); then
  printf '\nOpen: results/phase1/%s/selection_preview.jpg\n' "$EVENT"
  read -r -p "Is the GREEN box on the correct target, and the end-frame suitable? [y/N] " reply || true
  [[ "$reply" == [yY] || "$reply" == [yY][eE][sS] ]] || fail "Cancelled before any tracker runs; adjust --end-frame or investigate selection."
fi

for model in sam21 dam4sam; do
  dest="$EVENT_DIR/$model"
  if [[ -e "$dest/predictions.csv" || -e "$dest/evaluation_summary.json" ]] && (( ! FORCE )); then
    fail "$dest already contains experiment results. No overwrites by default. Review first or explicitly add --force."
  fi
done

info "STEP 2/5 — SAM2.1 Tiny ($EVENT, end=$END_FRAME)"
"$SAM_PY" src/run_phase1_event.py --event "$EVENT" --tracker sam21 --end-frame "$END_FRAME"
[[ -s "$EVENT_DIR/sam21/predictions.csv" && -s "$EVENT_DIR/sam21/run_metadata.json" ]] || fail "SAM2.1 outputs missing"

info "STEP 3/5 — DAM4SAM Tiny ($EVENT, end=$END_FRAME)"
"$DAM_PY" src/run_phase1_event.py --event "$EVENT" --tracker dam4sam --end-frame "$END_FRAME"
[[ -s "$EVENT_DIR/dam4sam/predictions.csv" && -s "$EVENT_DIR/dam4sam/run_metadata.json" ]] || fail "DAM4SAM outputs missing"

info "STEP 4/5 — Evaluate SAM2.1 and DAM4SAM (unchanged evaluator)"
"$SAM_PY" src/evaluation/event_evaluator.py --event "$EVENT" --tracker sam21
"$DAM_PY" src/evaluation/event_evaluator.py --event "$EVENT" --tracker dam4sam
for model in sam21 dam4sam; do
  [[ -s "$EVENT_DIR/$model/evaluation_frames.csv" && -s "$EVENT_DIR/$model/evaluation_summary.json" ]] || fail "$model evaluation outputs missing"
done

info "STEP 5/5 — Generate a comparison + frame-review images for THIS event only"
"$SAM_PY" - "$EVENT" "$ROOT" <<'PY'
import sys, csv, json, xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw
root = Path(sys.argv[2]); event_id = sys.argv[1]
sys.path.insert(0, str(root / 'src'))
from evaluation.event_loader import load_event

e = load_event(event_id)
base = root / 'results' / 'phase1' / event_id
manifest = json.loads((base / 'experiment_manifest.json').read_text())
start, end = manifest['selection_frame'], manifest['run_end_frame']
seq, target_id = e['sequence'], str(e['target_id'])
read_csv = lambda path: list(csv.DictReader(path.open(newline='')))
def is_true(value): return str(value).strip().lower() in ('true','1','yes')

preds, evaluations, metadata, summaries = {}, {}, {}, {}
for model in ('sam21', 'dam4sam'):
    model_dir = base / model
    preds[model] = {int(r['frame_number']):r for r in read_csv(model_dir/'predictions.csv')}
    evaluations[model] = {int(r['frame_number']):r for r in read_csv(model_dir/'evaluation_frames.csv')}
    summaries[model] = json.loads((model_dir/'evaluation_summary.json').read_text())
    metadata[model] = json.loads((model_dir/'run_metadata.json').read_text())
    m = metadata[model]
    if (m['event_id'] != event_id or m['tracker'] != model
            or int(m['selection_frame']) != start or int(m['run_end_frame']) != end
            or int(m['processed_future_frames']) != end-start):
        raise SystemExit(f'RUN CONFIG MISMATCH for {model}; do not compare these outputs')
    actual = set(preds[model]); expected = set(range(start+1, end+1))
    if actual != expected:
        raise SystemExit(f'{model} has missing/extra prediction frames: missing {len(expected-actual)}, extra {len(actual-expected)}')

comparison = {
    'event_id': event_id, 'sequence': seq, 'target_id': int(target_id),
    'selection_frame': start, 'run_end_frame': end,
    'interpretation_note': 'IoU against XML annotations is NOT visibility-aware; apparent recovery and possible-switch flags require manual video review.',
    'models': {}
}

# Select useful review frames. Keep a limited gallery per model.
frame_reasons = {}
def add(f, reason):
    if f is None: return
    f = int(f)
    if start <= f <= end:
        frame_reasons.setdefault(f, set()).add(reason)

add(start, 'selection')
for key in ('occlusion_start','maximum_occlusion_start','reappearance_frame','fully_visible_frame'):
    f = manifest.get(key)
    if f is not None:
        add(f, key)
        if key != 'fully_visible_frame':
            add(int(f)-1, f'before_{key}')
            add(int(f)+1, f'after_{key}')
add(end, 'last_frame')

for model in ('sam21','dam4sam'):
    p = preds[model]
    rows = [p[f] for f in sorted(p)]
    present = sum(is_true(r['present']) for r in rows)
    switches = [f for f,r in sorted(evaluations[model].items()) if is_true(r.get('possible_switch', False))]
    transitions = []
    previous = None
    for f, row in sorted(p.items()):
        now = is_true(row['present'])
        if previous is not None and now != previous:
            transitions.append((f, 'appeared' if now else 'absent'))
        previous = now
    for f, reason in transitions[:6]: add(f, f'{model}_{reason}')
    for f in switches[:4]: add(f, f'{model}_possible_switch')
    comparison['models'][model] = {
        'processed_future_frames': len(rows),
        'prediction_present_frames': present,
        'prediction_absent_frames': len(rows)-present,
        'target_matched_frames': summaries[model].get('target_matched_frames'),
        'target_annotated_frames': summaries[model].get('target_annotated_frames'),
        'possible_switch_frames': summaries[model].get('possible_switch_frames'),
        'average_tracker_runtime_ms': metadata[model].get('average_tracker_runtime_ms'),
        'first_presence_transitions': [{'frame':f,'state':state} for f,state in transitions[:10]],
        'possible_switch_frame_samples': switches[:10],
    }

# Include chronological landmarks first, and a limited number of event-specific flags.
landmarks = [f for f, reasons in sorted(frame_reasons.items())
             if any(x in reasons for x in ('selection','occlusion_start','maximum_occlusion_start',
                                            'reappearance_frame','fully_visible_frame','last_frame'))]
other = [f for f in sorted(frame_reasons) if f not in landmarks]
selected = sorted(set(landmarks + other[:max(0, 16-len(landmarks))]))[:16]
comparison['visual_review_frames'] = selected
(base/'comparison_summary.json').write_text(json.dumps(comparison, indent=2)+'\n')
with (base/'frame_review_candidates.csv').open('w', newline='') as out:
    writer = csv.writer(out); writer.writerow(['frame_number','reason','selected_for_gallery'])
    for f,reasons in sorted(frame_reasons.items()):
        writer.writerow([f, ';'.join(sorted(reasons)), f in selected])

# XML annotations are read once, and only requested frames are retained.
wanted = set(selected)
gt = {}
xml_root = ET.parse(e['xml_path']).getroot()
for frame in xml_root.findall('frame'):
    f = int(frame.get('num'))
    if f not in wanted: continue
    boxes = []
    for target in frame.findall('./target_list/target'):
        box = target.find('box')
        if box is None: continue
        x=float(box.get('left')); y=float(box.get('top'))
        w=float(box.get('width')); h=float(box.get('height'))
        boxes.append((target.get('id'), (x,y,x+w,y+h)))
    gt[f] = boxes

for model in ('sam21','dam4sam'):
    output = root/'results'/'visualizations'/f'{event_id}_{model}'
    output.mkdir(parents=True, exist_ok=True)
    for f in selected:
        img_path = Path(e['image_dir'])/f'img{f:05d}.jpg'
        if not img_path.is_file():
            raise SystemExit(f'Missing visualization frame {img_path}')
        img = Image.open(img_path).convert('RGB'); draw=ImageDraw.Draw(img)
        for tid, box in gt.get(f, []):
            color = '#1393ff' if tid == target_id else '#ff9900'
            draw.rectangle(box, outline=color, width=2)
            if tid == target_id:
                draw.text((box[0], max(0,box[1]-12)), f'GT target {tid}', fill=color,
                          stroke_width=1, stroke_fill='black')
        prediction = preds[model].get(f)
        if prediction and is_true(prediction['present']):
            try:
                box = tuple(float(prediction[k]) for k in ('box_x1','box_y1','box_x2','box_y2'))
                draw.rectangle(box, outline='#00ff00', width=3)
                draw.text((box[0],min(img.height-15,box[3]+2)),f'{model} prediction',
                          fill='#00ff00', stroke_width=1, stroke_fill='black')
            except (ValueError,TypeError):
                draw.text((10,25),'PREDICTION BOX MISSING/INVALID', fill='#ff4444')
        draw.text((8,6), f'{event_id} | {model} | frame {f}', fill='white', stroke_width=2, stroke_fill='black')
        img.save(output/f'frame_{f}.jpg', quality=92)

print(f'\nResults: {base}')
for model, result in comparison['models'].items():
    print(f'{model:7s}: matched {result["target_matched_frames"]} / annotated {result["target_annotated_frames"]}; '
          f'predictions present {result["prediction_present_frames"]}; '
          f'possible-switch flags {result["possible_switch_frames"]}; '
          f'avg {result["average_tracker_runtime_ms"]} ms')
print('Review frames:',selected)
print(f'Gallery: results/visualizations/{event_id}_sam21 and {event_id}_dam4sam')
print('Interpretation requires human review; no automatic recovery/identity verdict is claimed.')
PY

info "DONE: $EVENT ONLY. Send comparison_summary.json + evaluation_summary.json files and selected review images."
