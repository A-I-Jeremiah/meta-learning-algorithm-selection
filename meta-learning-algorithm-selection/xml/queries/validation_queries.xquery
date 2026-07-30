(: ===================================================================
   validation_queries.xquery
   ---------------------------------------------------------------
   Sanity / integrity queries over the experiment document. These check
   the data is internally consistent and summarise it for the manuscript.

   Target document: xml/samples/sample_experiment.xml
   Each query below is independent; run them separately.
   =================================================================== :)

(: --- Q1. Structural counts -------------------------------------- :)
<Summary>
  <datasets>{count(//Datasets/Dataset)}</datasets>
  <algorithms>{count(//Algorithms/Algorithm)}</algorithms>
  <runs>{count(//Runs/Run)}</runs>
  <expectedRuns>{count(//Datasets/Dataset)
                 * count(//Algorithms/Algorithm)
                 * xs:integer(//Experiment/@cvFolds)}</expectedRuns>
</Summary>

(: --- Q2. Referential integrity: any Run pointing at a dataset id
          that does not exist? (should be empty) ------------------ :)
//Runs/Run[not(@datasetRef = //Datasets/Dataset/@id)]

(: --- Q3. Value-range check: metric values outside plausible bounds.
          accuracy must be in [0,1]; R^2 <= 1. (should be empty) :)
//Run[(metric = 'accuracy' and (value < 0 or value > 1))
      or (metric = 'r2' and value > 1)]

(: --- Q4. Fold coverage: datasets that do NOT have exactly
          (algorithms x cvFolds) runs. (should be empty) ---------- :)
for $ds in //Datasets/Dataset
let $n := count(//Runs/Run[@datasetRef = $ds/@id])
let $expected := count(//Algorithms/Algorithm)
                 * xs:integer(//Experiment/@cvFolds)
where $n != $expected
return <Incomplete dataset="{$ds/@name}" runs="{$n}"
                   expected="{$expected}"/>

(: --- Q5. Mean accuracy per algorithm across all classification runs :)
<MeanAccuracyByAlgorithm>
{
  for $algo in //Algorithms/Algorithm
  let $vals := //Runs/Run[@algorithmRef = $algo/@id
                          and metric = 'accuracy']/value
  where count($vals) > 0
  order by avg($vals) descending
  return <Algorithm name="{$algo/@name}"
                    meanAccuracy="{avg($vals)}"
                    nRuns="{count($vals)}"/>
}
</MeanAccuracyByAlgorithm>
