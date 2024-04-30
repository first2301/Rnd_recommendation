from pycaret.anomaly import AnomalyExperiment
# import ray

# @ray.remote # 이상탐지의 경우, 분산학습을 하지 않아도 속도 차이가 없어서 ray 사용하지 않음
def train_caret_anomaly(updated_df):
    s = AnomalyExperiment()
    s.setup(data=updated_df, session_id=123, normalize_method = 'minmax')
    abod = s.create_model('abod')
    cluster = s.create_model('cluster')
    #cof = s.create_model('cof')
    iforest = s.create_model('iforest')
    histogram = s.create_model('histogram')
    knn = s.create_model('knn')
    lof = s.create_model('lof')
    #svm = s.create_model('svm')
    pca = s.create_model('pca')
    mcd = s.create_model('mcd')
    #sod = s.create_model('sod')
    # sos = s.create_model('sos')

    abod_anomalies = s.assign_model(abod)
    cluster_anomalies = s.assign_model(cluster)
    #cof_anomalies = s.assign_model(cof)
    iforest_anomalies = s.assign_model(iforest)
    histogram_anomalies = s.assign_model(histogram)
    knn_anomalies = s.assign_model(knn)
    lof_anomalies = s.assign_model(lof)
    #svm_anomalies = s.assign_model(svm)
    pca_anomalies = s.assign_model(pca)
    mcd_anomalies = s.assign_model(mcd)
    #sod_anomalies = s.assign_model(sod)
    # sos_anomalies = s.assign_model(sos)
    
    return {'abod_anomalies': abod_anomalies, 'cluster_anomalies': cluster_anomalies, 'iforest_anomalies': iforest_anomalies,
            'histogram_anomalies': histogram_anomalies, 'knn_anomalies': knn_anomalies, 'lof_anomalies': lof_anomalies,
            'pca_anomalies':pca_anomalies, 'mcd_anomalies': mcd_anomalies}

