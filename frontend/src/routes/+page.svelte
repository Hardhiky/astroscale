<script lang="ts">
	let inputs = {
		ra: 0,
		dec: 0,
		teff: 5000,
		logg: 4.5,
		fe_h: 0,
		snr: 50,
		parallax: 0.1
	};
	let prediction: number | null = null;
	let error = '';
	let loading = false;

	async function submit() {
		loading = true;
		error = '';
		prediction = null;

		try {
			const res = await fetch('http://localhost:8080/predict', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(inputs)
			});

			if (!res.ok) {
				throw new Error(`Server error: ${res.status}`);
			}

			const data = await res.json();
			prediction = data.predicted_z;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to get prediction';
			console.error('Prediction error:', e);
		} finally {
			loading = false;
		}
	}
</script>

<div class="mx-auto max-w-xl p-6">
	<h1 class="mb-4 text-2xl font-bold">Redshift Predictor</h1>
	<div class="grid gap-2">
		{#each Object.keys(inputs) as key}
			<label class="flex justify-between">
				{key}:
				<input type="number" step="any" bind:value={inputs[key]} class="w-32 rounded border p-1" />
			</label>
		{/each}
	</div>
	<button
		on:click={submit}
		disabled={loading}
		class="mt-4 rounded bg-blue-600 px-4 py-2 text-white disabled:bg-gray-400"
	>
		{loading ? 'Predicting...' : 'Predict'}
	</button>

	{#if error}
		<p class="mt-4 text-red-600">Error: {error}</p>
	{/if}

	{#if prediction !== null}
		<div class="mt-4 rounded border border-green-500 bg-green-50 p-4">
			<p class="text-lg font-semibold">Predicted Redshift (z): {prediction.toFixed(5)}</p>
		</div>
	{/if}
</div>
