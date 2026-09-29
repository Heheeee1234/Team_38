from clinical_data_gen.anomaly.features import PatientFeatureExtractor

extractor = PatientFeatureExtractor(history_window=6)

print("feature_count:", extractor.feature_count)
print("expected:", 67)