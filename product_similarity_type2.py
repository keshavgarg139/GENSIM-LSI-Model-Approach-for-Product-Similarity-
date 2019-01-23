from gensim import models,corpora,similarities
from importlib import reload
import mysql.connector
import csv
from statistics import mean
from nltk.corpus import stopwords
import sys
import numpy
import math
import re
reload(sys)

print("modules are imported")


def get_similarities(query,ans_list):
    s_lenth = len(ans_list)
    Corp = list(x.lower().split() for x in ans_list)

    dictionary = corpora.Dictionary(Corp)
    corpus = [dictionary.doc2bow(text) for text in Corp]

    lsi = models.LsiModel(corpus)
    corpus_lsi = lsi[corpus]

    vec_bow = dictionary.doc2bow(query.lower().split())
    vec_lsi = lsi[vec_bow]

    index = similarities.MatrixSimilarity(corpus_lsi)
    sims = index[vec_lsi]
    similarity = list(sims)

    end_lenth = len(similarity)
    if s_lenth != end_lenth:
        print('bug')
    return similarity

def remove_special_chars(text):
    final = [re.sub(r"[^a-zA-Z0-9]+", ' ', k) for k in text.split("\n")]
    return " ".join(final)

def remove_stopwords(sentence):
    stop = list(stopwords.words('english'))
    sent_split = list(y for y in sentence.split() if y not in stop)
    return ' '.join(sent_split)

def find_similar(required_topn,search_ids):

    connection = mysql.connector.connect(host='127.0.0.1',port=3341,database='wecat',user='developer',password='')
    db_Info = connection.get_server_info()
    print("Connected to MySQL database... MySQL Server version on ",db_Info)
    cursor = connection.cursor()

    similar_products = []

    print('\required_topn recieved by function are - ',required_topn,'\n')
    print('\ntype of required_topn recieved by function are - ',type(required_topn),'\n')

    for search_id in search_ids:

        print('\n id_product inside the core function is = ',search_id,'\n' )
        print('type of id_product is = ',type(search_id))  

        cmmd = "select id_product, product_title, attribute_key_1,attribute_value_1,attribute_key_2,attribute_value_2,attribute_key_3,attribute_value_3,attribute_key_4,attribute_value_4,attribute_key_5,attribute_value_5,feature_bullet_1,feature_bullet_2,feature_bullet_3,feature_bullet_4,feature_bullet_5,long_description from product_text_en where id_product in (select id_product from product where id_product_subtype = (select id_product_subtype from product where id_product = '{}') and id_product <> '{}' )".format(search_id,search_id)
        query = "select id_product, product_title, attribute_key_1,attribute_value_1,attribute_key_2,attribute_value_2,attribute_key_3,attribute_value_3,attribute_key_4,attribute_value_4,attribute_key_5,attribute_value_5,feature_bullet_1,feature_bullet_2,feature_bullet_3,feature_bullet_4,feature_bullet_5,long_description from product_text_en where id_product='%s'"%search_id

        metric_cmmd = "select id_product, product_length, product_height, product_width_depth from product_metric where id_product in (select id_product from product where id_product_subtype = (select id_product_subtype from product where id_product='{}') and id_product<>'{}')".format(search_id,search_id)
        query_metric = "select id_product, product_length, product_height, product_width_depth from product_metric where id_product = '%s'"%search_id

        brand_cmmd = "select id_product, id_brand from product where id_product in (select id_product from product where id_product_subtype = (select id_product_subtype from product where id_product='{}') and id_product<>'{}')".format(search_id,search_id)
        brand_query = "select id_product, id_brand from product where id_product = '%s' "%search_id

        print('cmmd = ',cmmd)
        print('query = ',query)

        cursor.execute(brand_cmmd)
        brand_cmmd_data = cursor.fetchall()

        cursor.execute(brand_query)
        query_brand = cursor.fetchall()
        query_brand = query_brand[0][1]

        cursor.execute(query)
        data = cursor.fetchall()
        query_data=[]

        query_title = remove_special_chars(data[0][1])
        query_title = remove_stopwords(query_title)

        query_desc = ' '.join(str(y) for y in data[0][2:] if y is not None )
        query_desc = remove_special_chars(query_desc)
        query_desc = remove_stopwords(query_desc)

        headers = ["sno", "id_product", "product_title", "description", "title_similarity", "desc_similarity","metrics_similarity", "brand_similarity", "total_similarity"]
        query_row_csv = ['Query', data[0][0], data[0][1], query_desc, 1,1,1,1,1]

        cursor.execute(query_metric)
        metric_query = cursor.fetchall()[0][1:]

        cursor.execute(metric_cmmd)
        metrics_data = cursor.fetchall()

        cursor.execute(cmmd)
        data = cursor.fetchall()
        print (type(data),'\n')
        print("lenght of data = ",len(data))

        topn = required_topn
        print("topn = ",topn)
        if len(data) < topn:
            topn = len(data)
        print("topn = ", topn)

        data2=[]
        data2.append(headers)
        data2.append(query_row_csv)

        title_list = []
        desc_list = []
        metric_similarities = []
        brand_similarities = []

        for i,row in enumerate(data) :
            new_row = []
            if row[1] is not None:
                new_row.append(i)
                new_row.append(row[0])
                new_row.append(row[1])
                title = row[1]
                title_list.append(remove_stopwords(remove_special_chars(title)))
            else:
                continue

            desc = ' '.join(str(y) for y in row[2:] if y is not None)
            new_row.append(desc)
            desc = remove_special_chars(desc)
            if desc is not '':
                desc_list.append(remove_stopwords(desc))
            else :
                desc_list.append(query_desc)

            metrics1 = list(float(x) if x is not None else x for x in metric_query)
            metrics2 = list(float(x) if x is not None else x for x in metrics_data[i][1:])
            flag = 1
            for i in range(len(metrics1)):
                if type(metrics1[i]) != type(metrics2[i]):
                    sim_metrics = 0
                    flag = 0
                    break
            if flag:
                metrics1 = list(float(0) if x is None else float(x) for x in metrics1)
                metrics2 = list(float(0) if x is None else float(x) for x in metrics2)
                if numpy.linalg.norm(metrics1) == 0 or numpy.linalg.norm(metrics2) == 0:
                    sim_metrics = 0
                else:
                    sim_metrics = numpy.dot(metrics1, metrics2) / (
                                numpy.linalg.norm(metrics1) * numpy.linalg.norm(metrics2))
                    sim_metrics = 0 if math.isnan(sim_metrics) else sim_metrics

            metric_similarities.append(sim_metrics)

            if brand_cmmd_data[i][1] is None:
                brand_sim = 1
            else:
                brand_sim = 1 if brand_cmmd_data[i][1] == query_brand else 0

            brand_similarities.append(brand_sim)

            data2.append(new_row)


        all_title_sims = get_similarities(query_title, title_list)
        all_desc_sims = get_similarities(query_desc, desc_list)

        all_sims = list(x for x in zip(all_title_sims,all_desc_sims,metric_similarities,brand_similarities))
        total_sims = []
        for x in all_sims:
            x = list(float(y) for y in x)
            total_sims.append(mean(x))

        for i,row in enumerate(data2[2:]):
            j=i+2
            data2[j] = data2[j] + list(all_sims[i])
            data2[j].append(total_sims[i])

        data2[2:] = sorted(data2[2:], key=lambda x: x[8], reverse=True)
        top = data2[:topn+2]

        similar_products.append(top)
        # file = '%s_newapproach.csv'%search_id
        # with open( file, 'w') as out:
        #     writer = csv.writer(out)
        #     for row in top:
        #         writer.writerow(row)

    if(connection.is_connected()):
        cursor.close()
        connection.close()
        print("MySQL connection is closed")
        
    return similar_products

def main():

    search_ids = ['N10987282A', 'N10987659A']
    prods = find_similar(20,search_ids)
    for i in prods:
        print(len(i))

if __name__== "__main__":
	main()