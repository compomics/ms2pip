# MS2PIP Server API
MS2PIP Server can be accessed through the webpage or through a RESTful API. While this page provides an example Python script for contacting the API, the Swagger-generated documentation can be found here: https://iomics.ugent.be/ms2pip/api/v2.

The following Python function contacts the MS2PIP Server RESTful API:

```python
def run_ms2pip_server(peptides, frag_method, ptm_list, url='https://iomics.ugent.be/ms2pip/api/v2'):
    # Check if all columns are present in dataframe
    for col in ['spec_id', 'peptide', 'charge', 'modifications']:
        if col not in peptides.columns:
            print("{} is missing from peptides DataFrame".format(col))
            return None

    # Split-up into batches of 100 000 peptides (maximum MS2PIP Server accepts per request)
    batch_size = 100000
    result = pd.DataFrame()
    for i in list(range(0, len(peptides), batch_size)):
        print("Working on batch {}/{}".format(int(i / batch_size + 1), len(peptides) // batch_size + 1))
        peptides_batch = peptides.iloc[i:i+batch_size, :]

        # Combine data into dictionary for json post request
        input_data = {
            "params": {
                "frag_method": frag_method,
                "ptm": ptm_list
            },
            "peptides": peptides_batch.to_dict(orient='list')
        }

        # Post data to server and get task id
        response = requests.post('{}/task'.format(url), json=input_data)
        if 'task_id' not in response.json():
            if 'error' in response.json():
                print("Server error: {}".format(response.json()['error']))
            else:
                print("Something went wrong")
            return None
        task_id = response.json()['task_id']
        print("Received task id: {}".format(task_id))

        # Check server task status and get result when ready
        sleep(1)
        response = requests.get('{}/task/{}/status'.format(url, task_id))
        state = response.json()['state']

        if state != 'SUCCESS':
            print("Check MS2PIP Server status every 5 seconds", end='')
            state = 'PENDING'
            pending_count = 0

            while state == 'PENDING' or state == 'PROGRESS':
                sleep(5)
                print('.', end='')
                response = requests.get('{}/task/{}/status'.format(url, task_id))
                state = response.json()['state']

                # Do not keep looping if task state is stuck on PENDING, it might have failed silently
                if state == 'PENDING':
                    pending_count += 1
                if pending_count > 24:
                    print("\nSomething went wrong")
                    return None
            print('')

        if state == 'SUCCESS':
            response = requests.get('{}/task/{}/result'.format(url, task_id))
            result_batch = pd.DataFrame.from_dict(response.json())['ms2pip_out']
            result_batch = result_batch[['spec_id', 'charge', 'ion', 'ionnumber', 'mz', 'prediction']]
            result = result.append(result_batch)
            print("Result received", end='\n\n')
        else:
            error_message = response.json()['status']
            print("Something went wrong: {}".format(error_message))
            return None

    print("Finished with all batches!")

    return result
```

The function takes three arguments: `peptides`, `frag_method` and `ptm_list`. `peptides` is an MS2PIP PEPREC-formatted Pandas DataFrame. `ptm_list` is a list of MS2PIP formatted PTM definitions. A detailed explanation of both data structures can be found on the [MS2PIP Server webpage](https://iomics.ugent.be/ms2pip/#howto). `frag_method` is the specific model with which you want to predict peak intensities (e.g. HCD, CID, iTRAQ...) Checkout [README.md](https://github.com/compomics/ms2pip_c/#ms2pip-models) for a list of all available models.

The function also takes an argument `url`, in which you can provide a custom URL to the server. By default this is `https://iomics.ugent.be/ms2pip/api`.

An example for running the function:
```python
# Define arguments
frag_method = 'HCD'
ptm_list = [
    'Oxidation,15.994915,M',
    'Carbamidomethyl,57.021464,C',
    'PhosphoS,79.966331,S',
    'PhosphoT,79.966331,T',
    'PhosphoY,79.966331,Y',
    'Glu->pyro-Glu,-18.010565,E',
    'Gln->pyro-Glu,-17.026549,Q',
    'Pyro-carbamidomethyl,39.994915,C',
    'Deamidated,0.984016,N',
    'iTRAQ,144.102063,N-term'
]
peptides = pd.DataFrame(data={
    'spec_id': ['peptide1', 'peptide2'],
    'peptide': ['STCINTFWLIVK', 'GRLNTFILK'],
    'modifications': ['3|Carbamidomethyl', '-'],
    'charge': [2, 3]
})

# Run function
result = run_ms2pip_server(peptides, frag_method, ptm_list)
```
