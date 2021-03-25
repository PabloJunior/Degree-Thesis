import numpy as np
import matplotlib.pyplot as plt
from math import pi, sinh
import time


N1 = N2 = 20;# количество разбиений расчетной сетки
eps=10**(-8);# допустимая погрешность устойчивости

As=1; # магнитный параметр
Diff=10**(-6); # коэффициенn диффузии


# нормы для погрешности
# m-норма/L∞
def inf_norm(matrix):
    m = len(matrix)
    n = len(matrix[0])
    
    a = []
    for i in range(m):
        row_abs = [abs(matrix[i][j]) for j in range(n)]
        a.append(sum(row_abs))
    return max(a);


# l-норма/L1
def one_norm(matrix):
    m = len(matrix)
    n = len(matrix[0])
    
    a = []
    for j in range(n):
        col_abs = [abs(matrix[i][j]) for i in range(m)]
        a.append(sum(col_abs))
    return max(a);


# напряженность магнитного поля ξ
def ksi(x1,x2): 
    return ((x1**2+x2**2+2)**2 - 8*x1**2)**(-1/4);

# функция b(x) = sinh(Aξ)/Aξ
def b(x1,x2):
    return sinh(As*ksi(x1,x2))/(As*ksi(x1,x2));


# значения расчетной области
# пространственные границы
# для x1 
A=-0.5;
B=0.5;

# для x2
C=0;
D=1;

l1=float(B-A);
l2=float(D-C);


# значения шагов
h1 = float(l1/N1);
h2 = float(l2/N2);
tau= h1**2;

# полушаги для функции b
q=0.5*h1;
w=0.5*h2;

# квадраты шагов для сокращения формул
a=h1**2;
s=h2**2;

x1= [i*h1 for i in range(0,N1+1)];
x2= [i*h2 for i in range(0,N2+1)];


#%% Заполнение матрицы A
A_matrix=np.zeros(( (N1+1)*(N2+1) , (N1+1)*(N2+1)));

# Угловые области

# 2.1 (i=0,j=0)
K=2*tau/b( x1[0],x2[0] )
# 0,0
A_matrix[0,0]= 1 + K*( b(x1[0]+q,x2[0])/a +  b(x1[0],x2[0]+w)/s );
# 0,1
A_matrix[0,1]= (-1)*K*( b(x1[0],x2[0]+w)/a )
# 1,0
A_matrix[0,N2+1]= (-1)*K*( b(x1[0]+q,x2[0])/s )


# 2.2 (i=0,j=N2)
K=2*tau/b( x1[0],x2[N2] )
# 0,N2
A_matrix[N2,N2]=  1 + K*( b(x1[0]+q,x2[N2])/a +  b(x1[0],x2[N2]-w)/s );
# 0,N2-1
A_matrix[N2,N2-1]= (-1)*K*( b(x1[0],x2[N2]-w)/s );
# 1,N2
A_matrix[N2,2*N2+1]= (-1)*K*( b(x1[0]+q,x2[N2])/a );


# 2.3 (i=N1,j=N2)
K=2*tau/b( x1[N1],x2[N2] )
# N1,N2
A_matrix[-1,-1]= 1 + K*( b(x1[N1]-q,x2[N2])/a +  b(x1[N1],x2[N2]-w)/s );
# N1,N2-1
A_matrix[-1,-2] = (-1)* K*( b(x1[N1],x2[N2]-w)/s );
# N1-1, N2
A_matrix[-1,-N2-2]= (-1)*K*( b(x1[N1]-q,x2[N2])/a );


# 2.4 (i=N1,j=0)
K= 2*tau/b( x1[N1],x2[0] )
# N1,0
A_matrix[-N2-1,-N2-1]= 1 + K*( b(x1[N1]-q,x2[0])/a +  b(x1[N1],x2[0]+w)/s );
# N1,1
A_matrix[-N2-1,-N2]= (-1)*K*( b(x1[N1],x2[0]+w)/s );
# N1-1,0
A_matrix[-N2-1,-2*N2-2]= (-1)*K*( b(x1[N1]-q,x2[0])/a );


# Граничные области

# 3.1 (i=0,j)
for i in range(1,N2):
    K=tau/b(x1[0], x2[i])
    # 0,j
    A_matrix[i,i]= 1 + K* ( 2*b(x1[0]+q, x2[i])/a + ( b(x1[0],x2[i]+w) + b(x1[0],x2[i]-w) )/s   )
    
    # 0,j-1
    A_matrix[i,i-1]= (-1)* K* ( ( b(x1[0],x2[i]-w) )/s   )

    # 0,j+1
    A_matrix[i,i+1]= (-1)*K* ( ( b(x1[0],x2[i]+w) )/s   )

    # 1,j
    A_matrix[i,i+N2+1]= (-1)*K*( 2*b(x1[0]+q, x2[i])/a  )


# 3.2 i,j=N2 
count=1;
for i in range((N1+1)*(N2+1)):
    if ((i!=N2) and (i!=(N1+1)*(N2+1)-1) and ((i+1)%(N2+1)==0) ):
        K=tau/b(x1[count],x2[N2])
        # i,N2
        A_matrix[i,i]= 1 +  K* ( ( b(x1[count]+q,x2[N2]) + b(x1[count]-q, x2[N2] ) )/a + 2*b(x1[count],x2[N2]-w)/s );
        
        # i, N2-1
        A_matrix[i,i-1]= (-1)*K* ( 2*b(x1[count],x2[N2]-w)/s );
        
        # i-1,N2
        A_matrix[i,i-N2-1] = (-1)*K* ( ( b(x1[count]-q, x2[N2] ) )/a );
        
        # i+1, N2
        A_matrix[i,i+N2+1] = (-1)*K* ( ( b(x1[count]+q,x2[N2]) )/a );
        
        count+=1;
 
    
# 3.3 i=N1,j
count=1;
for i in range((N1)*(N2+1)+1,(N1+1)*(N2+1)-1):
    K=tau/b(x1[N1],x2[count])
    # N1,j
    A_matrix[i,i]= 1 + K* ( 2*b(x1[N1]-q,x2[count])/a + ( b(x1[N1],x2[count]+w) + b(x1[N1], x2[count]-w ) )/s );

    # N1,i-1
    A_matrix[i,i-1]= (-1)*K* ( ( b(x1[N1], x2[count]-w ) )/s );
    
    # N1,i+1
    A_matrix[i,i+1]= (-1)*K* ( ( b(x1[N1], x2[count]+w ) )/s );

    # N1-1,j
    A_matrix[i,i-N2-1]=(-1)*K* ( 2*b(x1[N1]-q,x2[count])/a );

    count+=1;
        


# 3.4 (i,j=0)
count=1
for i in range((N1+1)*(N2+1)):
    if ((i!=0) and (i%(N2+1)==0) and (i!=N1*(N2+1))):
        K=tau/b( x1[count],x2[0] )
        # i,0
        A_matrix[i,i]= 1 +  K* ( ( b(x1[count]+q,x2[0]) + b(x1[count]-q,x2[0])  )/a  + 2*b(x1[count],x2[0]+w)/s )
        # i,1
        A_matrix[i,i+1]=(-1)* K*(2*b(x1[count],x2[0]+w)/s);
        # i-1,0
        A_matrix[i,i-N2-1]= (-1)*K* ( ( b(x1[count]-q,x2[0])  )/a  )
        # i+1,0
        A_matrix[i,i+N2+1] = (-1)*K* ( ( b(x1[count]+q,x2[0])  )/a  )
        count+=1;

        
# Внутренняя область    
k1=1;
for i in range(1,N1):
    k2=1;
    for j in range(i*N2+1+i,(i+1)*N2+i):
        K=tau/b( x1[k1], x2[k2] )
        # i,j
        A_matrix[j,j] = 1 +  K* ( ( b(x1[k1]+q, x2[k2]) + b(x1[k1]-q, x2[k2]) )/a + ( b(x1[k1], x2[k2]+w) + b(x1[k1], x2[k2]-w) )/s )
        
        # i,j-1
        A_matrix[j,j-1] = (-1)* K * ( ( b(x1[k1], x2[k2]-w) )/s )
        
        # i,j+1
        A_matrix[j,j+1] = (-1)* K* ( ( b(x1[k1], x2[k2]+w) )/s )
        
        # i-1,j
        A_matrix[j,j-N2-1]= (-1)*K* ( ( b(x1[k1]-q, x2[k2]) )/a )
        
        # i+1,j
        A_matrix[j,j+N2+1]= (-1)*K* ( ( b(x1[k1]+q, x2[k2]) )/a )
        
        k2+=1;

    k1+=1;
    

#%% построение численного решения
    
# численное решение промежуточной задачи (3.1)
u = np.zeros((1,N1+1,N2+1));

# численное решение исходной задачи (1.7)
C_num=np.full((1,N1+1,N2+1),1);


# 1-ый шаг алгоритма
# заполняю нулевой временной слой согласно начальному условию
for i in range(0,N1+1):
    for j in range(0,N2+1):
        u[0,i,j]=1/b(x1[i],x2[j]);


p=1; # счётчик количества итераций
start_time = time.time() # счётчик времени итераций

while(p>0):
    prev=u[p-1,:,:].copy();
    Cj=C_num[p-1,:,:].copy();

    # определение решения СЛАУ Ax=b
    B_list = [ y for x in prev for y in x];
    answer = np.linalg.solve(A_matrix,B_list);
    
    # заполнение следующего временного слоя
    new = np.reshape(answer,(N1+1,N2+1));
    u = np.vstack((u,new[None]))   
    
    Cj_1=np.zeros((N1+1,N2+1));
    for i in range(0,N1+1):
        for j in range(0,N2+1):
            Cj_1[i,j]=b(x1[i],x2[j])*new[i,j];       
    C_num = np.vstack((C_num,Cj_1[None]));   

    if (inf_norm(Cj_1-Cj)<eps):
        break;
    p+=1;

    
end_time = time.time()

print("Количество временных слоёв для точности eps = "+str(eps)+" составляет "+ str(p));
print("Время, потребовавшееся для этого компьютеру, составляет " + str(round(end_time-start_time,3)) + " секунд")

# определение примерного времени протекания процесса    
realt=p*tau*(l1**2)/Diff;
def convert(seconds): 
    min, sec = divmod(seconds, 60) 
    hour, min = divmod(min, 60)
    if (hour<24):
        return "%d часов, %02d минут и %02d секунд" % (hour, min, sec)
    else:
        day,hour=divmod(hour, 24)
        return "%d суток, %02d часов, %02d минут и %02d секунд" % (day,hour, min, sec)


print("Время, потребовавшееся для этого, составляет " + str(round(realt)) + " секунд или " + convert(realt))
print()    


#%% определение аналитического решения, демонстрация графиков
           
# аналитическое решение задачи по формуле (2.3)
C_exact=np.zeros((N1+1,N2+1));
  
# приближенное вычисление интеграла по формуле Симпсона          
I=I=(B-A)*(D-C)/36 * (b(A,C) + b(A,D)+ b(B,C)+ b(B,D) + 4*(b(A,0.5*(C+D)) + b(B,0.5*(C+D)) + b(0.5*(A+B),C) +b(0.5*(A+B),D) ) +\
16*(b(0.5*(A+B),0.5*(C+D) )) );

for i in range(0,N1+1):
    for j in range(0,N2+1):
        C_exact[i,j]=b(x1[i],x2[j])*(I**(-1));
        
C_exact=(B-A)*(D-C)*C_exact;

# погрешность вычислений
Z=C_num[p,:,:]-C_exact;
print("Норма погрешности: "+str(one_norm(Z)))


# просмотреть графики при фиксированном значении x_1
for i in range(0,N1+1,5):
    plt.figure();
    plt.plot(x2,C_num[p,i,:],'r*', label='Численное');
    plt.plot(x2,C_exact[i,:],'bD', label='Аналитическое')
    plt.title('Решения при x1 = ' + str(x1[i]));
    plt.legend()
    plt.xlabel("x2")
    plt.ylabel("C")
    plt.show()

# просмотреть графики при фиксированном значении x_2
for j in range(0,N2+1,5):
    plt.figure();
    plt.plot(x1,C_num[p,:,j],'rx', label='Численное')
    plt.plot(x1,C_exact[:,j],'bD', label='Аналитическое')
    plt.title('Решения при x2 = ' + str(x2[j]));
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("C")
    plt.show()
