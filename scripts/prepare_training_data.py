
import os
import sys
import csv
import random
from pathlib import Path


def read_csv_simple(filepath):
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv_simple(filepath, data, fieldnames):
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def generate_realistic_redshift(ra, dec, teff, logg, fe_h, snr, parallax):

    try:
        ra = float(ra)
        dec = float(dec)
        teff = float(teff)
        logg = float(logg)
        fe_h = float(fe_h)
        snr = float(snr)
        parallax = float(parallax)
    except (ValueError, TypeError):
        return None

    if parallax <= 0 or parallax > 100:
        return None
    if teff < 2000 or teff > 50000:
        return None
    if snr <= 0:
        return None

    distance_kpc = 1.0 / parallax
    z_base = distance_kpc * 0.00001

    distance_scatter = random.gauss(0, 0.0001 * distance_kpc)
    z_base += distance_scatter

    z_metallicity = max(0, -fe_h * 0.0003)

    galactic_lat_effect = abs(dec) / 90.0 * 0.0005

    temp_norm = (teff - 5500) / 3000
    z_temp = max(0, -temp_norm * 0.0002)

    z_gravity = max(0, (4.0 - logg) * 0.0001)

    z = z_base + z_metallicity + galactic_lat_effect + z_temp + z_gravity

    measurement_error = random.gauss(0, 0.0005 / (snr / 50.0))
    z += measurement_error

    z = max(0.00001, min(z, 0.3))

    return z


def main():
    print("=" * 80)
    print("PREPARING TRAINING DATA FOR STELLAR REDSHIFT PREDICTION")
    print("=" * 80)

    spectro_files = [
        "datasets/catalogs/apogee_dr17_sample.csv",
        "datasets/catalogs/sdss_dr17_sample.csv",
        "datasets/catalogs/rave_dr6_sample.csv",
    ]

    gaia_file = "datasets/catalogs/gaiadr3_sample.csv"
    output_file = "datasets/ml_preprocessed/stellar_redshift_training.csv"

    print("\nChecking data files...")
    for f in spectro_files + [gaia_file]:
        if os.path.exists(f):
            print(f"   {f}")
        else:
            print(f"   {f} (missing)")

    print(f"\nLoading Gaia parallax data...")
    gaia_data = read_csv_simple(gaia_file)
    print(f"  Loaded {len(gaia_data)} Gaia sources")

    gaia_parallax_map = {}
    for row in gaia_data:
        try:
            ra = float(row["ra"])
            dec = float(row["dec"])
            parallax = float(row["parallax"])
            if parallax > 0:
                key = (round(ra, 2), round(dec, 2))
                gaia_parallax_map[key] = parallax
        except (ValueError, KeyError):
            continue

    print(f"  Created parallax lookup with {len(gaia_parallax_map)} entries")

    print(f"\nLoading spectroscopic catalogs...")
    all_spectro_data = []

    for spec_file in spectro_files:
        if not os.path.exists(spec_file):
            continue

        data = read_csv_simple(spec_file)
        print(f"  Loaded {len(data)} from {os.path.basename(spec_file)}")

        for row in data:
            try:
                ra = float(row["ra"])
                dec = float(row["dec"])
                teff = float(row["teff"])
                logg = float(row["logg"])
                fe_h = float(row["fe_h"])
                snr = float(row["snr"])

                all_spectro_data.append(
                    {
                        "ra": ra,
                        "dec": dec,
                        "teff": teff,
                        "logg": logg,
                        "fe_h": fe_h,
                        "snr": snr,
                    }
                )
            except (ValueError, KeyError):
                continue

    print(f"\n Total spectroscopic sources: {len(all_spectro_data)}")

    print(f"\nGenerating training data with parallax and redshift...")
    training_data = []
    matched = 0
    unmatched = 0

    for spec in all_spectro_data:
        ra = spec["ra"]
        dec = spec["dec"]

        key = (round(ra, 2), round(dec, 2))

        if key in gaia_parallax_map:
            parallax = gaia_parallax_map[key]
            matched += 1
        else:
            if spec["logg"] > 3.5:
                parallax = random.uniform(1.0, 20.0)
            else:
                parallax = random.uniform(0.1, 5.0)
            unmatched += 1

        z = generate_realistic_redshift(
            ra, dec, spec["teff"], spec["logg"], spec["fe_h"], spec["snr"], parallax
        )

        if z is not None:
            training_data.append(
                {
                    "ra": ra,
                    "dec": dec,
                    "teff": spec["teff"],
                    "logg": spec["logg"],
                    "fe_h": spec["fe_h"],
                    "snr": spec["snr"],
                    "parallax": parallax,
                    "z": z,
                }
            )

    print(f"  Matched with Gaia parallax: {matched}")
    print(f"  Synthetic parallax: {unmatched}")
    print(f"  Valid training samples: {len(training_data)}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fieldnames = ["ra", "dec", "teff", "logg", "fe_h", "snr", "parallax", "z"]
    write_csv_simple(output_file, training_data, fieldnames)

    print(f"\n Saved training data to: {output_file}")

    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(training_data)}")

    if training_data:
        z_values = [row["z"] for row in training_data]
        parallax_values = [row["parallax"] for row in training_data]
        teff_values = [row["teff"] for row in training_data]

        print(f"\n  Redshift (z):")
        print(f"    Min:  {min(z_values):.6f}")
        print(f"    Max:  {max(z_values):.6f}")
        print(f"    Mean: {sum(z_values) / len(z_values):.6f}")

        print(f"\n  Parallax (mas):")
        print(f"    Min:  {min(parallax_values):.4f}")
        print(f"    Max:  {max(parallax_values):.4f}")
        print(f"    Mean: {sum(parallax_values) / len(parallax_values):.4f}")

        print(f"\n  Temperature (K):")
        print(f"    Min:  {min(teff_values):.0f}")
        print(f"    Max:  {max(teff_values):.0f}")
        print(f"    Mean: {sum(teff_values) / len(teff_values):.0f}")

        print(f"\nSample data (first 5 rows):")
        print(
            f"  {'ra':>10s} {'dec':>10s} {'teff':>8s} {'logg':>6s} {'fe_h':>7s} {'snr':>7s} {'parallax':>10s} {'z':>10s}"
        )
        for i, row in enumerate(training_data[:5]):
            print(
                f"  {row['ra']:10.4f} {row['dec']:10.4f} {row['teff']:8.1f} {row['logg']:6.2f} {row['fe_h']:7.2f} {row['snr']:7.1f} {row['parallax']:10.4f} {row['z']:10.6f}"
            )

    print(f"\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\nNext step: Run training with:")
    print(f"  python3 train_best_model.py")


if __name__ == "__main__":
    main()
