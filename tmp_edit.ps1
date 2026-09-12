 = Get-Content -Path 'studio/frontend/src/features/export/constants.ts' -Raw
 = @''

export type MergeMethodType =  linear | ties | dare_ties | ctm;

export const MERGE_METHODS: { value: MergeMethodType; label: string; description: string }[] = [
  { value: linear, label: Linear, description: Simple weighted average of adapter weights. },
  { value: ties, label: TIES, description: Trim Elect Interpolate Sign - resolves conflicting weight signs. },
  { value: dare_ties, label: DARE-TIES, description: Dropout-aware TIES - randomly drops small deltas before merging. },
  { value: ctm, label: CtM, description: Compressed Target Merge - low-rank SVD compression after merging. },
];

''@ + 'export const GUIDE_STEPS'
 =  -replace '(?m)^export const GUIDE_STEPS', 
Set-Content -Path 'studio/frontend/src/features/export/constants.ts' -Value  -NoNewline
Write-Host 'constants.ts updated'