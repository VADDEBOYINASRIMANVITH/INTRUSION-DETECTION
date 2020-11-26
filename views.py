import numpy as np
import pandas as pd

from django.shortcuts import render

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

kdd_train = pd.read_csv('C:/Users/Ammulu/PycharmProjects/IntrusionDetection/App/train70.csv', header=None)
kdd_train.head()

kdd_test = pd.read_csv('C:/Users/Ammulu/PycharmProjects/IntrusionDetection/App/test30.csv', header=None)
kdd_test.head()

kdd_train.columns = ['train_duration', 'train_protocol_type', 'train_service', 'train_flag', 'src_bytes',
                     'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                     'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                     'num_shells', 'num_access_files', 'dummy', 'num_outbound_cmds', 'is_host_login',
                     'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                     'srv_rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                     'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                     'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                     'dst_host_srv_serror_rate', 'train_dst_host_rerror_rate', 'dst_bushost_srv_rerror_rate']

kdd_test.columns = ['test_duration', 'test_protocol_type', 'test_service', 'test_flag', 'src_bytes', 'dst_bytes',
                    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                    'num_access_files', 'dummy', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
                    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                    'dst_host_srv_serror_rate', 'test_dst_host_rerror_rate', 'dst_bushost_srv_rerror_rate']

kdd_train.head()
kdd_test.head()

# Dropping Unessasry coloumns
kdd_train_clean = kdd_train.drop(
    ['wrong_fragment', 'urgent', 'num_failed_logins', 'num_file_creations', 'num_shells', 'dummy',
     'num_outbound_cmds'], axis=1)

kdd_test_clean = kdd_test.drop(
    ['wrong_fragment', 'urgent', 'num_failed_logins', 'num_file_creations', 'num_shells', 'dummy',
     'num_outbound_cmds'], axis=1)

kdd_train_clean.head()
kdd_test_clean.head()

# Checking the datatypes of the coloumns
kdd_train_clean.info()
kdd_test_clean.info()

# Basic Statistics of columns
kdd_train_clean.describe()
kdd_test_clean.describe()

# counting the categories of columns
kdd_train_clean['train_protocol_type'].value_counts()
kdd_test_clean['test_protocol_type'].value_counts()

kdd_train_clean['train_service'].value_counts()
kdd_test_clean['test_service'].value_counts()

kdd_train_clean['train_flag'].value_counts()
kdd_test_clean['test_flag'].value_counts()

kdd_train_clean['train_dst_host_rerror_rate'].value_counts()
kdd_test_clean['test_dst_host_rerror_rate'].value_counts()

# # Data Transormation
train_protocol_type = {'tcp': 0, 'udp': 1, 'icmp': 2}
train_protocol_type.items()
kdd_train_clean.train_protocol_type = [train_protocol_type[item] for item in kdd_train_clean.train_protocol_type]
kdd_train_clean.head(20)

test_protocol_type = {'tcp': 0, 'udp': 1, 'icmp': 2}
test_protocol_type.items()
kdd_test_clean.test_protocol_type = [test_protocol_type[item] for item in kdd_test_clean.test_protocol_type]
kdd_test_clean.head(20)

# Checking the condition and data transformation
train_duration = kdd_train_clean['train_duration']
for i in train_duration:
    if i <= 2:
        print('good condition', i)
    else:
        print('bad condition', i)

kdd_train_clean['train_duration'] = np.where((kdd_train_clean.train_duration <= 2), 0, 1)
kdd_train_clean.head(20)

test_duration = kdd_test_clean['test_duration']
for i in test_duration:
    if i <= 2:
        print('good condition', i)
    else:
        print('bad condition', i)
kdd_test_clean['test_duration'] = np.where((kdd_test_clean.test_duration <= 2), 0, 1)
kdd_test_clean.head(20)


train_replace_map = {'normal': "normal", 'DOS': ['back', 'land', 'pod', 'neptune', 'smurf', 'teardrop'],
                     'R2L': ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'spy', 'phf', 'warezclient',
                             'warezmaster'],
                     'U2R': ['buffer_overflow', 'loadmodule', 'perl', 'rootkit'],
                     'PROBE': ['ipsweep', 'nmap', 'portsweep', 'satan']}

kdd_train_format = kdd_train_clean.assign(
    train_dst_host_rerror_rate=kdd_train_clean['train_dst_host_rerror_rate'].apply(
        lambda x: [key for key, value in train_replace_map.items() if x in value][0]))
kdd_train_format.head(20)


test_replace_map = {'normal': "normal",
                    'DOS': ['back', 'land', 'pod', 'neptune', 'smurf', 'teardrop'],
                    'R2L': ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'spy', 'phf', 'warezclient',
                            'warezmaster'],
                    'U2R': ['buffer_overflow', 'loadmodule', 'perl', 'rootkit'],
                    'PROBE': ['ipsweep', 'nmap', 'portsweep', 'satan'],
                    'extra new attacks': ['apache2', 'httptunnel', 'mailbomb', 'mscan', 'named', 'processtable',
                                          'ps', 'saint', 'sendmail', 'snmpgetattack', 'snmpguess', 'sqlattack',
                                          'udpstorm', 'worm', 'xlock', 'xsnoop', 'xterm']}

kdd_test_format = kdd_test_clean.assign(test_dst_host_rerror_rate=kdd_test_clean['test_dst_host_rerror_rate'].apply(
    lambda x: [key for key, value in test_replace_map.items() if x in value][0]))
kdd_test_format.head(20)

train_service = {'aol': 1, 'auth': 2, 'bgp': 3, 'courier': 4, 'csnet_ns': 5, 'ctf': 6, 'daytime': 7, 'discard': 8,
                 'domain': 9, 'domain_u': 10, 'echo': 11, 'eco_i': 12, 'ecr_i': 13, 'efs': 14, 'exec': 15,
                 'finger': 16, 'ftp': 17, 'ftp_data': 18, 'gopher': 19, 'harvest': 20, 'hostnames': 21, 'http': 22,
                 'http_2784': 23, 'http_443': 24, 'http_8001': 25, 'imap4': 26, 'IRC': 27, 'iso_tsap': 28,
                 'klogin': 29, 'kshell': 30, 'ldap': 31, 'link': 32, 'login': 33, 'mtp': 34, 'name': 35,
                 'netbios_dgm': 36, 'netbios_ns': 37, 'netbios_ssn': 38, 'netstat': 39, 'nnsp': 40, 'nntp': 41,
                 'ntp_u': 42, 'other': 43, 'pm_dump': 44, 'pop_2': 45, 'pop_3': 46, 'printer': 47, 'private': 48,
                 'red_i': 49, 'remote_job': 50, 'rje': 51, 'shell': 52, 'smtp': 53, 'sql_net': 54, 'ssh': 55,
                 'sunrpc': 56, 'supdup': 57, 'systat': 58, 'telnet': 59, 'tftp_u': 60, 'tim_i': 61, 'time': 62,
                 'urh_i': 63, 'urp_i': 64, 'uucp': 65, 'uucp_path': 66, 'vmnet': 67, 'whois': 68, 'X11': 69,
                 'Z39_50': 70}
train_service.items()

kdd_train_format.train_service = [train_service[item] for item in kdd_train_format.train_service]
kdd_train_format.head(20)


test_service = {'auth': 1, 'bgp': 2, 'courier': 3, 'csnet_ns': 4, 'ctf': 5, 'daytime': 6, 'discard': 7, 'domain': 8,
                'domain_u': 9, 'echo': 10, 'eco_i': 11, 'ecr_i': 12, 'efs': 13, 'exec': 14, 'finger': 15, 'ftp': 16,
                'ftp_data': 17, 'gopher': 18, 'hostnames': 19, 'http': 20, 'http_443': 21, 'imap4': 22, 'IRC': 23,
                'iso_tsap': 24, 'klogin': 25, 'kshell': 26, 'ldap': 27, 'link': 28, 'login': 29, 'mtp': 30,
                'name': 31, 'netbios_dgm': 3, 'netbios_ns': 33, 'netbios_ssn': 34, 'netstat': 35, 'nnsp': 36,
                'nntp': 37, 'ntp_u': 38, 'other': 39, 'pm_dump': 40, 'pop_2': 41, 'pop_3': 42, 'printer': 43,
                'private': 44, 'remote_job': 45, 'rje': 46, 'shell': 47, 'smtp': 48, 'sql_net': 49, 'ssh': 50,
                'sunrpc': 51, 'supdup': 52, 'systat': 53, 'telnet': 54, 'tftp_u': 55, 'tim_i': 56, 'time': 57,
                'urp_i': 58, 'uucp': 59, 'uucp_path': 60, 'vmnet': 61, 'whois': 62, 'X11': 63, 'Z39_50': 64,'urh_i':65,
                'harvest':66,'red_i':67}
test_service.items()

kdd_test_format.test_service = [test_service[item] for item in kdd_test_format.test_service]
kdd_test_format.head(20)

train_dst_host_rerror_rate = {'normal': 0, 'DOS': 1, 'R2L': 2, 'U2R': 3, 'PROBE': 4}
train_dst_host_rerror_rate.items()
kdd_train_format.train_dst_host_rerror_rate = [train_dst_host_rerror_rate[item] for item in
                                               kdd_train_format.train_dst_host_rerror_rate]
kdd_train_format.head(20)

test_dst_host_rerror_rate = {'normal': 0, 'DOS': 1, 'R2L': 2, 'U2R': 3, 'PROBE': 4, 'extra new attacks': 5}
test_dst_host_rerror_rate.items()
kdd_test_format.test_dst_host_rerror_rate = [test_dst_host_rerror_rate[item] for item in
                                             kdd_test_format.test_dst_host_rerror_rate]
kdd_test_format.head(20)

train_flag = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'S1': 5, 'SH': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9,
              'OTH': 10}

train_flag.items()
kdd_train_format.train_flag = [train_flag[item] for item in kdd_train_format.train_flag]
kdd_train_format.head(20)

test_flag = {'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'S1': 5, 'SH': 6, 'S2': 7, 'RSTOS0': 8, 'S3': 9,
             'OTH': 10}
kdd_test_format.test_flag = [test_flag[item] for item in kdd_test_format.test_flag]
kdd_test_format.head(20)



#############################
train_corr = kdd_train_format.corr()
test_corr = kdd_test_format.corr()
print(train_corr)
print(test_corr)

y_train = kdd_train_format.iloc[:, -2].values
X = kdd_train_format.drop(['train_dst_host_rerror_rate'], axis=1)
X_train = X.iloc[:, :].values

# Two features with highest chi-squared statistics are selected
# chi2_features = SelectKBest(chi2, k = 10)
# x_train_best = chi2_features.fit_transform(x1, y1)


y_test = kdd_test_format.iloc[:, -2].values
Z = kdd_test_format.drop(['test_dst_host_rerror_rate'], axis=1)
X_test = Z.iloc[:, :].values


# Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# applying pca
pca = PCA(n_components=19)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# Create your views here.
def home(request):
    return render(request, 'home.html')

def result(request):
    return render(request, 'result.html')


#visualising confusion matrix

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



# # Logistic  regression


def Lr(request):
    # Actual Lr

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    data = {'accuracy_lr': metrics.accuracy_score(y_test, y_pred_lr),
            'cm_lr': confusion_matrix(y_test, y_pred_lr),
            'mse_lr': metrics.mean_squared_error(y_test, y_pred_lr),
            'mae_lr': metrics.mean_absolute_error(y_test, y_pred_lr),
            'rmse_lr': np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)),
            'rsquared_lr': metrics.r2_score(y_test,y_pred_lr)
            }

    return render(request, 'Lr.html', data)

    # # Support vector machine


def Svm(request):
    # actual svc
    model = SVC(kernel='rbf', C=1, gamma=1)
    model.fit(X_train, y_train)
    y_pred_svm = model.predict(X_test)
    #plot_confusion_matrix_svm(cm_svm, classes=['0', '1', '2', '3', '4', '5'], normalize=False, title='Confusion matrix')
    #plt.savefig(r"C:\Users\Ammulu\PycharmProjects\IntrusionDetection\App\static\images\svm.png")
    Overalldata = {'accuracy_svm': metrics.accuracy_score(y_test, y_pred_svm),
            'cm_svm': confusion_matrix(y_test, y_pred_svm),
            'mse_svm': metrics.mean_squared_error(y_test, y_pred_svm),
            'mae_svm': metrics.mean_absolute_error(y_test, y_pred_svm),
            'rmse_svm': np.sqrt(metrics.mean_squared_error(y_test, y_pred_svm)),
            'rsquared_svm': metrics.r2_score(y_test,y_pred_svm)
            }

    return render(request, 'Svm.html', Overalldata)

    # # Random Forest


def Rf(request):
    # actual Rf
    regressor = RandomForestClassifier(n_estimators=500)
    regressor.fit(X_train, y_train)
    y_pred_rf = regressor.predict(X_test)
    #plot_confusion_matrix_rf(cm_rf, classes=['0', '1', '2', '3', '4', '5'], normalize=False, title='Confusion matrix')
    #plt.savefig(r"C:\Users\Ammulu\PycharmProjects\IntrusionDetection\App\static\images\randomforest.png")


    data = {'accuracy_rf': metrics.accuracy_score(y_test, y_pred_rf),
            'cm_rf': confusion_matrix(y_test, y_pred_rf),
            'mse_rf': metrics.mean_squared_error(y_test, y_pred_rf),
            'mae_rf': metrics.mean_absolute_error(y_test, y_pred_rf),
            'rmse_rf': np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)),
            'rsquared_rf': metrics.r2_score(y_test,y_pred_rf)
            }

    return render(request, 'Rf.html', data)



