python condense.py --reproduce -d cifar10 -f 2 --ipc 10  --cluster_method Kmeans --weight True
python condense.py --reproduce -d cifar10 -f 2 --ipc 10  --cluster_method Kmeans++ --weight True
python condense.py --reproduce -d cifar10 -f 2 --ipc 10  --cluster_method DBSCAN --weight True
python condense.py --reproduce -d cifar10 -f 2 --ipc 10  --cluster_method Kmedoids --weight True
python condense.py --reproduce -d cifar10 -f 2 --ipc 10  --cluster_method Agglomerative --weight False
python condense.py --reproduce -d cifar10 -f 2 --ipc 10  --cluster_method Birch --weight True





# if Time allowed
python condense.py --reproduce -d cifar10 -f 2 --ipc 10  --cluster_method Kmeans++ --weight False
python condense.py --reproduce -d cifar10 -f 2 --ipc 10  --cluster_method DBSCAN --weight False
python condense.py --reproduce -d cifar10 -f 2 --ipc 10  --cluster_method Kmedoids --weight False
python condense.py --reproduce -d cifar10 -f 2 --ipc 10  --cluster_method Birch --weight False