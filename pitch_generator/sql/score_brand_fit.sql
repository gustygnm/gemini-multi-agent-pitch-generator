CREATE OR REPLACE TABLE bwg.brand_review AS
SELECT
  campaign,
  concept,
  brand_fit,
  IF(brand_fit >= 7, 'on brand', 'needs another pass') AS verdict
FROM (
  SELECT
    campaign,
    concept,
    AI.SCORE((
      "Rate this campaign key visual from 1 to 10 against our house brand guidelines:\n"
      "- Palette: deep indigo and slate, with one warm accent of amber or terracotta\n"
      "- Lighting: single low raking light source with long shadows\n"
      "- Composition: subject at the off-center third, generous empty negative space\n"
      "- Subject: one clear realistic photographic subject, shallow depth of field\n"
      " Penalize heavily for neon or fluorescent colors, text or watermarks, busy "
      "backgrounds, multiple competing subjects, or flat even lighting.\n"
      "Campaign concept: ", concept, " ", key_visual),
      connection_id => 'us-central1.pitch-connection',
      endpoint => 'gemini-3.8-flash'
    ) AS brand_fit
  FROM bwg.key_visuals
);
