from flask import Flask, render_template, redirect, url_for, request
import product_similarity_type2 as pst2
import csv
import os

app = Flask(__name__)

@app.route('/')
def homepage() :
	return(render_template('get_query_product_id.html'))


@app.route('/get_similars/',methods = ['POST'])
def call_find_similar():
	data = request.form.to_dict()
	id_product = data['id_product']
	required_topn = data['topn']
	required_topn = int(required_topn)
	print('\nrequired_topn in app function is = ',required_topn,'\n')
	print('\ntype of required_topn is = ',type(required_topn),'\n')
	lst1 = []
	lst1.append(id_product)
	print('list being sent to the fucntion is - ',lst1,'\n')
	similars = pst2.find_similar(required_topn,lst1)

	return render_template('display_similars.html',similars = similars)

if __name__ == '__main__':
	app.debug = True
	app.run()
	app.run(debug = True)
