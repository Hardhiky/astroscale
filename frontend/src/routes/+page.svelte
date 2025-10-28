<script lang="ts">
	import { onMount } from 'svelte';

	// Stellar parameters with better defaults
	let inputs = {
		ra: 200.12,
		dec: -47.33,
		teff: 5800,
		logg: 4.3,
		fe_h: 0.0,
		snr: 100,
		parallax: 7.2
	};

	// Field metadata with proper descriptions
	const fieldInfo: Record<
		string,
		{ label: string; unit: string; min: number; max: number; step: number; description: string }
	> = {
		ra: {
			label: 'Right Ascension',
			unit: 'degrees',
			min: 0,
			max: 360,
			step: 0.01,
			description: 'Angular distance along the celestial equator'
		},
		dec: {
			label: 'Declination',
			unit: 'degrees',
			min: -90,
			max: 90,
			step: 0.01,
			description: 'Angular distance north or south of celestial equator'
		},
		teff: {
			label: 'Effective Temperature',
			unit: 'K',
			min: 2000,
			max: 50000,
			step: 100,
			description: 'Surface temperature of the star'
		},
		logg: {
			label: 'Surface Gravity',
			unit: 'log(cm/s²)',
			min: 0,
			max: 5,
			step: 0.1,
			description: 'Logarithm of surface gravity'
		},
		fe_h: {
			label: 'Metallicity [Fe/H]',
			unit: 'dex',
			min: -3,
			max: 0.5,
			step: 0.1,
			description: 'Iron abundance relative to solar'
		},
		snr: {
			label: 'Signal-to-Noise Ratio',
			unit: 'dimensionless',
			min: 1,
			max: 500,
			step: 1,
			description: 'Quality of spectroscopic measurement'
		},
		parallax: {
			label: 'Parallax',
			unit: 'mas',
			min: 0.01,
			max: 100,
			step: 0.01,
			description: 'Apparent shift in position (distance indicator)'
		}
	};

	// Preset stellar configurations
	const presets = {
		'Sun-like Star': {
			ra: 200.12,
			dec: -47.33,
			teff: 5800,
			logg: 4.3,
			fe_h: 0.0,
			snr: 100,
			parallax: 7.2,
			description: 'G-type main sequence star similar to our Sun',
			color: '#FDB813'
		},
		'Hot Blue Star': {
			ra: 85.3,
			dec: -15.8,
			teff: 8500,
			logg: 4.0,
			fe_h: 0.2,
			snr: 60,
			parallax: 1.2,
			description: 'Hot, massive A-type star',
			color: '#9BB0FF'
		},
		'Red Giant': {
			ra: 150.5,
			dec: 30.2,
			teff: 4200,
			logg: 2.5,
			fe_h: -0.5,
			snr: 80,
			parallax: 3.5,
			description: 'Evolved star with expanded outer layers',
			color: '#FF6B6B'
		},
		'Metal-poor Star': {
			ra: 310.7,
			dec: 60.4,
			teff: 5500,
			logg: 4.5,
			fe_h: -2.0,
			snr: 40,
			parallax: 5.8,
			description: 'Old, low metallicity Population II star',
			color: '#CC99FF'
		},
		'White Dwarf': {
			ra: 45.9,
			dec: -70.1,
			teff: 12000,
			logg: 8.0,
			fe_h: -0.3,
			snr: 25,
			parallax: 15.0,
			description: 'Dense stellar remnant',
			color: '#E0E0E0'
		}
	};

	// State management
	let prediction: number | null = null;
	let error = '';
	let loading = false;
	let lastRequest: typeof inputs | null = null;
	let selectedPreset: string | null = null;
	let showInfo = false;
	let predictionHistory: Array<{ inputs: typeof inputs; z: number; timestamp: number }> = [];

	// Submit prediction request
	async function submit() {
		loading = true;
		error = '';
		prediction = null;
		lastRequest = null;

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
			lastRequest = { ...inputs };

			// Add to history
			predictionHistory = [
				{ inputs: { ...inputs }, z: prediction, timestamp: Date.now() },
				...predictionHistory.slice(0, 4)
			];
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to get prediction';
			console.error('Prediction error:', e);
		} finally {
			loading = false;
		}
	}

	// Load preset
	function loadPreset(presetName: string) {
		const preset = presets[presetName];
		inputs = {
			ra: preset.ra,
			dec: preset.dec,
			teff: preset.teff,
			logg: preset.logg,
			fe_h: preset.fe_h,
			snr: preset.snr,
			parallax: preset.parallax
		};
		selectedPreset = presetName;
		prediction = null;
		error = '';
		lastRequest = null;
	}

	// Get interpretation of redshift value
	function getRedshiftInterpretation(z: number): {
		category: string;
		description: string;
		distance: string;
	} {
		if (z < 0.0001) {
			return {
				category: 'Very Nearby',
				description: 'Within our immediate stellar neighborhood',
				distance: '< 100 parsecs'
			};
		} else if (z < 0.001) {
			return {
				category: 'Nearby',
				description: 'Within the Milky Way galaxy',
				distance: '100-1000 parsecs'
			};
		} else if (z < 0.01) {
			return {
				category: 'Distant',
				description: 'Outer regions of our galaxy or nearby galaxies',
				distance: '1-10 kiloparsecs'
			};
		} else if (z < 0.1) {
			return {
				category: 'Very Distant',
				description: 'Nearby galaxy clusters',
				distance: '> 10 kiloparsecs'
			};
		} else {
			return {
				category: 'Extremely Distant',
				description: 'Distant galaxies',
				distance: '> 100 kiloparsecs'
			};
		}
	}

	// Visualize redshift on spectrum
	function getRedshiftPosition(z: number): number {
		// Log scale for better visualization
		const logZ = Math.log10(Math.max(z, 0.00001));
		const minLog = -5; // 0.00001
		const maxLog = -1; // 0.1
		return ((logZ - minLog) / (maxLog - minLog)) * 100;
	}

	// Get color for star type based on temperature
	function getStarColor(teff: number): string {
		if (teff < 3700) return '#FF6B6B'; // Red
		if (teff < 5200) return '#FFA500'; // Orange
		if (teff < 6000) return '#FDB813'; // Yellow
		if (teff < 7500) return '#F0F0F0'; // White
		if (teff < 10000) return '#9BB0FF'; // Blue-white
		return '#5B8EFF'; // Blue
	}
</script>

<svelte:head>
	<title>Stellar Redshift Predictor - AstroScale</title>
</svelte:head>

<div class="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950">
	<!-- Header -->
	<header class="border-b border-blue-900/30 bg-slate-900/50 backdrop-blur-sm">
		<div class="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
			<div class="flex items-center justify-between">
				<div>
					<h1 class="text-3xl font-bold text-white">
						<span class="text-blue-400">Astro</span>Scale
					</h1>
					<p class="mt-1 text-sm text-blue-200/70">Stellar Redshift Prediction System</p>
				</div>
				<button
					on:click={() => (showInfo = !showInfo)}
					class="rounded-lg border border-blue-500/30 bg-blue-500/10 px-4 py-2 text-sm text-blue-300 transition-colors hover:bg-blue-500/20"
				>
					{showInfo ? 'Hide' : 'Show'} Info
				</button>
			</div>
		</div>
	</header>

	<main class="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
		<!-- Info Panel -->
		{#if showInfo}
			<div
				class="mb-8 rounded-xl border border-blue-500/30 bg-slate-900/80 p-6 backdrop-blur-sm"
				style="animation: fadeIn 0.3s ease-in-out"
			>
				<h2 class="mb-3 text-xl font-semibold text-white">About Redshift</h2>
				<div class="space-y-2 text-sm text-blue-200/80">
					<p>
						<strong class="text-blue-300">Redshift (z)</strong> is a measure of how much light from a
						celestial object has been stretched to longer wavelengths due to the object's motion or distance.
					</p>
					<p>
						This tool predicts stellar redshift based on spectroscopic parameters including
						temperature, surface gravity, metallicity, and distance indicators.
					</p>
					<p class="text-xs text-blue-300/60">
						Model: Gradient Boosting • Training: 19,000 stars • Accuracy: R² = 0.30
					</p>
				</div>
			</div>
		{/if}

		<div class="grid grid-cols-1 gap-8 lg:grid-cols-3">
			<!-- Left Column: Input Controls -->
			<div class="lg:col-span-2">
				<!-- Preset Buttons -->
				<div class="mb-6 rounded-xl border border-blue-500/30 bg-slate-900/50 p-6 backdrop-blur-sm">
					<h2 class="mb-4 text-lg font-semibold text-white">Quick Presets</h2>
					<div class="grid grid-cols-2 gap-3 sm:grid-cols-3">
						{#each Object.entries(presets) as [name, preset]}
							<button
								on:click={() => loadPreset(name)}
								class="group relative overflow-hidden rounded-lg border border-blue-500/30 bg-slate-800/50 p-4 text-left transition-all hover:border-blue-400/50 hover:bg-slate-800/80"
								class:ring-2={selectedPreset === name}
								class:ring-blue-500={selectedPreset === name}
							>
								<div class="mb-2 flex items-center gap-2">
									<div
										class="h-3 w-3 rounded-full shadow-lg"
										style="background: {preset.color}; box-shadow: 0 0 10px {preset.color}50"
									></div>
									<span class="text-sm font-medium text-white">{name}</span>
								</div>
								<p class="text-xs text-blue-200/60">{preset.description}</p>
							</button>
						{/each}
					</div>
				</div>

				<!-- Parameter Inputs -->
				<div class="rounded-xl border border-blue-500/30 bg-slate-900/50 p-6 backdrop-blur-sm">
					<h2 class="mb-4 text-lg font-semibold text-white">Stellar Parameters</h2>
					<div class="grid gap-4 sm:grid-cols-2">
						{#each Object.entries(inputs) as [key, value]}
							<div class="group">
								<label class="mb-2 block text-sm font-medium text-blue-200">
									{fieldInfo[key].label}
									<span class="text-xs text-blue-300/50">({fieldInfo[key].unit})</span>
								</label>
								<input
									type="number"
									bind:value={inputs[key]}
									min={fieldInfo[key].min}
									max={fieldInfo[key].max}
									step={fieldInfo[key].step}
									class="w-full rounded-lg border border-blue-500/30 bg-slate-800/50 px-4 py-2.5 text-white placeholder-blue-300/30 transition-all focus:border-blue-400 focus:ring-2 focus:ring-blue-500/20 focus:outline-none"
								/>
								<p class="mt-1 text-xs text-blue-300/50">{fieldInfo[key].description}</p>
							</div>
						{/each}
					</div>

					<!-- Submit Button -->
					<button
						on:click={submit}
						disabled={loading}
						class="mt-6 w-full rounded-lg bg-gradient-to-r from-blue-600 to-blue-500 px-6 py-3 font-semibold text-white shadow-lg shadow-blue-500/30 transition-all hover:from-blue-500 hover:to-blue-400 hover:shadow-blue-500/50 disabled:from-slate-600 disabled:to-slate-600 disabled:shadow-none"
					>
						{#if loading}
							<span class="flex items-center justify-center gap-2">
								<svg class="h-5 w-5 animate-spin" viewBox="0 0 24 24">
									<circle
										class="opacity-25"
										cx="12"
										cy="12"
										r="10"
										stroke="currentColor"
										stroke-width="4"
										fill="none"
									></circle>
									<path
										class="opacity-75"
										fill="currentColor"
										d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
									></path>
								</svg>
								Calculating...
							</span>
						{:else}
							Predict Redshift
						{/if}
					</button>
				</div>
			</div>

			<!-- Right Column: Results & Visualization -->
			<div class="space-y-6">
				<!-- Error Display -->
				{#if error}
					<div
						class="rounded-xl border border-red-500/30 bg-red-500/10 p-4 backdrop-blur-sm"
						style="animation: fadeIn 0.3s ease-in-out"
					>
						<p class="text-sm font-medium text-red-300">Error: {error}</p>
					</div>
				{/if}

				<!-- Prediction Result -->
				{#if prediction !== null && lastRequest}
					<div
						class="rounded-xl border border-emerald-500/30 bg-gradient-to-br from-emerald-500/10 to-blue-500/10 p-6 backdrop-blur-sm"
						style="animation: fadeIn 0.5s ease-in-out"
					>
						<h2 class="mb-4 text-sm font-medium text-emerald-300">Prediction Result</h2>

						<!-- Main Z Value -->
						<div class="mb-6">
							<div class="mb-2 text-5xl font-bold text-white">
								z = {prediction.toFixed(6)}
							</div>
							<div
								class="inline-block rounded-full px-3 py-1 text-xs font-medium"
								style="background: {getStarColor(lastRequest.teff)}20; color: {getStarColor(
									lastRequest.teff
								)}"
							>
								{getRedshiftInterpretation(prediction).category}
							</div>
						</div>

						<!-- Redshift Spectrum Visualization -->
						<div class="mb-6">
							<p class="mb-2 text-xs font-medium text-blue-200">Redshift Spectrum</p>
							<div
								class="relative h-12 rounded-lg bg-gradient-to-r from-blue-600 via-purple-500 to-red-500 p-1"
							>
								<div
									class="absolute top-1/2 h-8 w-1 -translate-y-1/2 bg-white shadow-lg"
									style="left: {getRedshiftPosition(
										prediction
									)}%; box-shadow: 0 0 20px rgba(255,255,255,0.8)"
								></div>
							</div>
							<div class="mt-1 flex justify-between text-xs text-blue-300/60">
								<span>0.00001</span>
								<span>0.0001</span>
								<span>0.001</span>
								<span>0.01</span>
								<span>0.1</span>
							</div>
						</div>

						<!-- Interpretation -->
						<div class="rounded-lg border border-blue-500/30 bg-slate-900/50 p-4">
							<p class="mb-1 text-sm font-medium text-white">
								{getRedshiftInterpretation(prediction).description}
							</p>
							<p class="text-xs text-blue-300/70">
								Estimated Distance: {getRedshiftInterpretation(prediction).distance}
							</p>
						</div>

						<!-- Star Visualization -->
						<div class="mt-4 flex items-center justify-center py-4">
							<div
								class="relative h-20 w-20 rounded-full"
								style="background: radial-gradient(circle, {getStarColor(
									lastRequest.teff
								)} 0%, {getStarColor(
									lastRequest.teff
								)}50 50%, transparent 100%); box-shadow: 0 0 40px {getStarColor(
									lastRequest.teff
								)}80"
							>
								<div class="absolute inset-0 animate-pulse rounded-full bg-white opacity-30"></div>
							</div>
						</div>
					</div>
				{/if}

				<!-- Prediction History -->
				{#if predictionHistory.length > 0}
					<div class="rounded-xl border border-blue-500/30 bg-slate-900/50 p-4 backdrop-blur-sm">
						<h3 class="mb-3 text-sm font-medium text-white">Recent Predictions</h3>
						<div class="space-y-2">
							{#each predictionHistory as record}
								<div
									class="flex items-center justify-between rounded-lg bg-slate-800/50 p-3 text-xs"
								>
									<div class="flex items-center gap-2">
										<div
											class="h-2 w-2 rounded-full"
											style="background: {getStarColor(record.inputs.teff)}"
										></div>
										<span class="text-blue-200">T={record.inputs.teff}K</span>
									</div>
									<span class="font-mono text-emerald-300">z = {record.z.toFixed(6)}</span>
								</div>
							{/each}
						</div>
					</div>
				{/if}
			</div>
		</div>
	</main>

	<!-- Footer -->
	<footer class="mt-16 border-t border-blue-900/30 bg-slate-900/50 py-6">
		<div class="mx-auto max-w-7xl px-4 text-center text-sm text-blue-200/50">
			<p>AstroScale Stellar Redshift Predictor • Powered by Machine Learning</p>
		</div>
	</footer>
</div>

<style>
	@keyframes fadeIn {
		from {
			opacity: 0;
			transform: translateY(10px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}

	input[type='number']::-webkit-inner-spin-button,
	input[type='number']::-webkit-outer-spin-button {
		-webkit-appearance: none;
		margin: 0;
	}

	input[type='number'] {
		-moz-appearance: textfield;
	}
</style>
