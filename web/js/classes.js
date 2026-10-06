/** classes.js — mirrors src/utils/classes.py (single source of truth lives in Python). */
export const CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
  'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'];
export const GROUPS = { person: [0, 1], vehicle: [3, 4, 5, 8], 'two-wheeler': [2, 9], tricycle: [6, 7] };
export const GROUP_ORDER = ['person', 'vehicle', 'two-wheeler', 'tricycle'];
export const GROUP_LABELS = { person: 'Persons', vehicle: 'Vehicles', 'two-wheeler': 'Two-wheelers', tricycle: 'Tricycles' };
export const GROUP_COLORS = { person: '#1e78eb', vehicle: '#3cc850', 'two-wheeler': '#ff9614', tricycle: '#aa50c8' };
const CLASS_TO_GROUP = new Map();
for (const [g, cs] of Object.entries(GROUPS)) for (const c of cs) CLASS_TO_GROUP.set(c, g);
export function groupOf(cls) {
  const g = CLASS_TO_GROUP.get(cls);
  if (!g) throw new Error(`unknown VisDrone class index: ${cls}`);
  return g;
}
