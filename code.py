#Swiss roll and S-curve datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import pairwise_distances
#Full Gaussian Kernel Diffusion map
def diffusion_map_full(X,n_components=3,epsilon=None):
    D=pairwise_distances(X)
    if epsilon is None:
        epsilon=np.median(D)**2
    K=np.exp(-(D**2)/epsilon)
    P=K/K.sum(axis=1,keepdims=True)
    eigvals,eigvecs=np.linalg.eig(P)
    eigvals,eigvecs=eigvals.real,eigvecs.real
    idx=np.argsort(-eigvals)
    return eigvals[idx][1:n_components+1],eigvecs[:,idx][:,1:n_components+1]
#KNN Diffusion Maps
def diffusion_map_knn(X,n_components=3,k=10,epsilon=None):
    nbrs=NearestNeighbors(n_neighbors=k).fit(X)
    dist,idxs=nbrs.kneighbors(X)
    if epsilon is None:
        epsilon=np.median(dist)**2
    N=len(X)
    K=np.zeros((N,N))
    for i in range(N):
        for j,d in zip(idxs[i],dist[i]):
            K[i,j]=np.exp(-(d**2)/epsilon)
    P=K/K.sum(axis=1,keepdims=True)
    eigvals,eigvecs=np.linalg.eig(P)
    eigvals,eigvecs=eigvals.real,eigvecs.real
    idx=np.argsort(-eigvals)
    return eigvals[idx][1:n_components+1],eigvecs[:,idx][:,1:n_components+1]
#PCA embedding
def pca_embedding(X,n_components=2):
    pca=PCA(n_components=n_components)
    coords=pca.fit_transform(X)
    return coords,pca.explained_variance_ratio_
X_swiss,t_swiss=make_swiss_roll(2000,noise=0.05)
X_scurve,t_scurve=make_s_curve(2000,noise=0.05)
y_swiss=(t_swiss>np.median(t_swiss)).astype(int)
y_scurve=(t_scurve>np.median(t_scurve)).astype(int)
scaler=StandardScaler()
X_swiss_std=scaler.fit_transform(X_swiss)
X_scurve_std=scaler.fit_transform(X_scurve)
#Data Visualization
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.scatter(X_swiss[:,0],X_swiss[:,2],c=t_swiss,cmap='Spectral')
plt.title("Swiss Roll")
plt.subplot(1,2,2)
plt.scatter(X_scurve[:,0],X_scurve[:,2],c=t_scurve,cmap='Spectral')
plt.title("S-Curve")
plt.show()
vals_swiss_full,vecs_swiss_full=diffusion_map_full(X_swiss_std,3)
vals_scurve_full,vecs_scurve_full=diffusion_map_full(X_scurve_std,3)
vals_swiss_knn,vecs_swiss_knn=diffusion_map_knn(X_swiss_std,3,k=12)
vals_scurve_knn,vecs_scurve_knn=diffusion_map_knn(X_scurve_std,3,k=12)
pca_swiss,var_swiss=pca_embedding(X_swiss_std,2)
pca_scurve,var_scurve=pca_embedding(X_scurve_std,2)
def plot_embedding(coords,color,title):
    plt.figure(figsize=(6,5))
    plt.scatter(coords[:,0],coords[:,1],c=color,cmap='Spectral')
    plt.title(title)
    plt.show()
plot_embedding(vecs_swiss_full[:,:2],t_swiss,"Swiss Roll – Diffusion Map")
plot_embedding(vecs_swiss_knn[:,:2],t_swiss,"Swiss Roll – Diffusion Map (KNN)")
plot_embedding(pca_swiss,t_swiss,"Swiss Roll – PCA")
plot_embedding(vecs_scurve_full[:,:2],t_scurve,"S-Curve – Diffusion Map")
plot_embedding(vecs_scurve_knn[:,:2],t_scurve,"S-Curve – Diffusion Map (KNN)")
plot_embedding(pca_scurve,t_scurve,"S-Curve – PCA")
#Confusion matrices and time graphs
def plot_confusion(cm,title):
    plt.figure(figsize=(4,4))
    sns.heatmap(cm,annot=True,fmt='d',cmap="viridis")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
def plot_time_bars(names,train_times,test_times,title_prefix):
    plt.figure(figsize=(6,4))
    plt.bar(names,train_times)
    plt.title(f"{title_prefix} – Training Time")
    plt.show()
    plt.figure(figsize=(6,4))
    plt.bar(names,test_times)
    plt.title(f"{title_prefix} – Prediction Time")
    plt.show()
def evaluate_all(name,X,y,emb_raw,emb_pca,emb_dm,emb_knn):
    methods=["Raw 2D","PCA 2D","DM Full 2D","DM KNN 2D"]
    embeddings=[emb_raw,emb_pca,emb_dm,emb_knn]
    tr_idx,te_idx=train_test_split(np.arange(len(y)),test_size=0.2,stratify=y,random_state=42)
    cms,train_times,test_times,accuracies=[],[],[],[]
    for emb in embeddings:
        Xtr,Xte=emb[tr_idx],emb[te_idx]
        ytr,yte=y[tr_idx],y[te_idx]
        clf=KNeighborsClassifier(5)
        t0=time.time()
        clf.fit(Xtr,ytr)
        train_times.append(time.time()-t0)
        t0=time.time()
        pred=clf.predict(Xte)
        test_times.append(time.time()-t0)
        cms.append(confusion_matrix(yte,pred))
        accuracies.append(accuracy_score(yte,pred))
    for m,cm in zip(methods,cms):
        plot_confusion(cm,f"{name} — {m}")
    plot_time_bars(methods,train_times,test_times,name)
    print(f"\n=== {name} — 5-FOLD CV ===")
    skf=StratifiedKFold(5,shuffle=True,random_state=42)
    for m,emb in zip(methods,embeddings):
        scores=[]
        for tr,te in skf.split(emb,y):
            model=KNeighborsClassifier(5)
            model.fit(emb[tr],y[tr])
            scores.append(accuracy_score(y[te],model.predict(emb[te])))
        print(f"{m}: {np.mean(scores):.4f}")
evaluate_all("Swiss Roll",X_swiss,y_swiss,X_swiss_std[:,:2],pca_swiss,vecs_swiss_full[:,:2],vecs_swiss_knn[:,:2])
evaluate_all("S-Curve",X_scurve,y_scurve,X_scurve_std[:,:2],pca_scurve,vecs_scurve_full[:,:2],vecs_scurve_knn[:,:2])
#Breast cancer dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import make_swiss_roll,make_s_curve,load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import pairwise_distances
RANDOM_STATE=42
def diffusion_maps(X,n_components=3,k=None,alpha=0.5):
    N=X.shape[0]
    D=pairwise_distances(X)
    eps=np.median(D[D>0])**2*5.0
    if k is None:
        K=np.exp(-D**2/eps)
    else:
        K=np.zeros((N,N))
        knn=np.argsort(D,axis=1)[:,1:k+1]
        for i in range(N):
            for j in knn[i]:
                v=np.exp(-(D[i,j]**2)/eps)
                K[i,j]=v
                K[j,i]=v
    q=K.sum(axis=1)
    q=np.maximum(q,1e-12)
    q_alpha=q**(-alpha)
    K_tilde=K*(q_alpha[:,None]@q_alpha[None,:])
    row_sum=K_tilde.sum(axis=1,keepdims=True)
    P=K_tilde/row_sum
    eigvals,eigvecs=np.linalg.eig(P)
    eigvals=eigvals.real
    eigvecs=eigvecs.real
    idx=np.argsort(-eigvals)
    eigvals=eigvals[idx]
    eigvecs=eigvecs[:,idx]
    eigvecs/=np.linalg.norm(eigvecs,axis=0,keepdims=True)
    return eigvals[1:n_components+1],eigvecs[:,1:n_components+1]
def plot_scatter(X2,y,title):
    plt.figure(figsize=(6,5))
    plt.scatter(X2[:,0],X2[:,1],c=y,cmap="coolwarm",s=14)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()
def plot_cm(cm,title):
    plt.figure(figsize=(4,4))
    sns.heatmap(cm,annot=True,fmt="d",cmap="viridis",xticklabels=[0,1],yticklabels=[0,1])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
def plot_accuracy(methods,accuracies,name):
    plt.figure(figsize=(6,4))
    plt.bar(methods,accuracies)
    for i,a in enumerate(accuracies):
        plt.text(i,a,f"{a:.3f}",ha='center',va='bottom')
    plt.ylim(0,1)
    plt.title(f"{name} – Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.show()
def plot_times(methods,train_times,test_times,name):
    plt.figure(figsize=(6,4))
    plt.bar(methods,train_times)
    plt.title(f"{name} – kNN Training Time")
    plt.ylabel("Seconds")
    plt.show()
    plt.figure(figsize=(6,4))
    plt.bar(methods,test_times)
    plt.title(f"{name} – kNN Testing Time")
    plt.ylabel("Seconds")
    plt.show()
def plot_psi(psi,title):
    plt.figure(figsize=(5,4))
    plt.scatter(psi,psi[::-1],c=psi,cmap="viridis",s=12)
    plt.colorbar(label="ψ value")
    plt.title(title)
    plt.xlabel("ψ")
    plt.ylabel("Mirrored ψ")
    plt.tight_layout()
    plt.show()
def plot_eigs(eigvals,title):
    plt.figure(figsize=(6,4))
    plt.plot(eigvals[:10],marker="o")
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()
def evaluate_dataset(name,X,y):
    print(f"\n\n==================== {name} ====================")
    Xs=StandardScaler().fit_transform(X)
    pca=PCA(n_components=2)
    X_pca=pca.fit_transform(Xs)
    _,X_dm_full=diffusion_maps(Xs,n_components=2,k=None)
    _,X_dm_knn=diffusion_maps(Xs,n_components=2,k=12)
    plot_scatter(X_pca,y,f"{name} – PCA (2D)")
    plot_scatter(X_dm_full,y,f"{name} – Diffusion Maps FULL (2D)")
    plot_scatter(X_dm_knn,y,f"{name} – Diffusion Maps kNN (2D)")
    methods=["PCA","DM Full","DM kNN"]
    embeddings=[X_pca,X_dm_full,X_dm_knn]
    accuracies=[]
    train_times=[]
    test_times=[]
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)
    for Z,mname in zip(embeddings,methods):
        fold_acc=[]
        fold_train=[]
        fold_test=[]
        final_cm=None
        for tr,te in skf.split(Z,y):
            Xtr,Xte=Z[tr],Z[te]
            ytr,yte=y[tr],y[te]
            model=KNeighborsClassifier(n_neighbors=5)
            t0=time.time()
            model.fit(Xtr,ytr)
            fold_train.append(time.time()-t0)
            t0=time.time()
            pred=model.predict(Xte)
            fold_test.append(time.time()-t0)
            fold_acc.append(accuracy_score(yte,pred))
            final_cm=confusion_matrix(yte,pred)
        plot_cm(final_cm,f"{name} – {mname}")
        accuracies.append(np.mean(fold_acc))
        train_times.append(np.mean(fold_train))
        test_times.append(np.mean(fold_test))
    plot_accuracy(methods,accuracies,name)
    plot_times(methods,train_times,test_times,name)
    return Xs
if __name__=="__main__":
    data=load_breast_cancer()
    Xb=data.data
    yb=data.target
    Xb_std=evaluate_dataset("Breast Cancer",Xb,yb)
    eigvals,eigvecs=diffusion_maps(Xb_std,n_components=5)
    plot_psi(eigvecs[:,0],"Breast Cancer – ψ1")
    plot_psi(eigvecs[:,1],"Breast Cancer – ψ2")
    plot_psi(eigvecs[:,2],"Breast Cancer – ψ3")
    plot_eigs(eigvals,"Breast Cancer – Diffusion Maps Eigenvalue Spectrum")

