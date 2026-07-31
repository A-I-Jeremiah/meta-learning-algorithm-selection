(: ===================================================================
   best_algorithm_per_dataset.xquery
   ---------------------------------------------------------------
   The brief's required XQuery:
     "for each dataset return the algorithm with the highest average rank."

   We interpret "highest average rank" as best mean performance: for each
   dataset, average each algorithm's per-fold values, then pick the algorithm
   with the greatest mean (accuracy or R^2 -- higher is better in both).

   Target document: xml/samples/sample_experiment.xml
   Returns a <BestAlgorithms> tree, one <Dataset> per benchmark.
   =================================================================== :)

<BestAlgorithms>
{
  for $ds in //Datasets/Dataset
  let $runs := //Runs/Run[@datasetRef = $ds/@id]
  (: mean value per algorithm on this dataset :)
  let $means :=
    for $algo in distinct-values($runs/@algorithmRef)
    let $vals := $runs[@algorithmRef = $algo]/value
    return
      <AlgoMean algorithmRef="{$algo}" mean="{avg($vals)}"/>
  (: the algorithm whose mean is the maximum :)
  let $best := $means[xs:double(@mean) = max($means/xs:double(@mean))][1]
  order by $ds/@name
  return
    <Dataset name="{$ds/@name}" task="{$ds/@task}">
      <BestAlgorithm ref="{$best/@algorithmRef}"
                     meanScore="{$best/@mean}"/>
    </Dataset>
}
</BestAlgorithms>
