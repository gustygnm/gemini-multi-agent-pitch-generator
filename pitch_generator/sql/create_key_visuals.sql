CREATE OR REPLACE TABLE bwg.key_visuals AS
SELECT
  campaign,
  concept,
  OBJ.FETCH_METADATA(
    OBJ.MAKE_REF(uri, 'us-central1.pitch-connection')
  ) AS key_visual
FROM UNNEST([
  STRUCT('cats' AS campaign, 'Flying skateboards for cats' AS concept,
         'gs://local-dev-project-bwg/key-visuals/cats.png' AS uri),
  STRUCT('bike', 'A commuter bike built for rainy cities',
         'gs://local-dev-project-bwg/key-visuals/bike.png')
]);
