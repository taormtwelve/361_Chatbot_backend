from flask import Flask, jsonify, request
from firebase_admin import credentials, firestore, initialize_app
from flask_cors import CORS
import numpy as np
from keras.models import load_model
from pythainlp import word_vector, word_tokenize
from pythainlp.tag.named_entity import ThaiNameTagger
import random, csv,os
from pythainlp.tag import pos_tag
from datetime import date
from fuzzywuzzy import fuzz

# _________________________________________ initial _________________________________________

cred = credentials.Certificate("chatbot-1c5b9-firebase-adminsdk-ls39w-48a67efdb9.json")
initialize_app(cred)
db = firestore.client()
chatbot_ref = db.collection('chatbot')
QnA_ref = db.collection('QnA')
frequencyQ_ref = db.collection('frequencyQ')
app = Flask(__name__)
cors = CORS(app)

# _____________________________________ Machine Learning _____________________________________

model = load_model('./models/chatbot_best_val_2.h5')
category_0 = ['ความเป็นมาประวัติก่อ', 'อายุปีคอมพิวเตอร์คอม', 'อายุปีสารสนเทศเครือข่าย', 'รุ่นคอมพิวเตอร์คอม', 'รุ่นสารสนเทศเครือข่าย']
category_2 = ['ภาควิชาภาคเมเจอร์', 'อาจารย์ครู']
category_3 = ['หัวหน้า', 'รองหัวหน้า', 'คนท่าน', 'อาจารย์ครู']
category_5 = ['บัณฑิตปริญญาตรี', 'มหาบัณฑิตปริญญาโท', 'ดุษฎีบัณฑิตปริญญาเอก', 'ปีระยะเวลา', 'ค่าเทอมค่าธรรมเนียม']
f_questions = ['ความเป็นมาของภาควิชา', 'สถานที่ตั้งของภาควิชา', 'ช่องทางการติดต่อ', 'อาจารย์', 'ข่าวที่น่าสนใจที่เกี่ยวกับภาควิชา',
               'หลักสูตรการศึกษา', 'เรียนเกี่ยวกับอะไรบ้าง', 'เกณฑ์การรับนักศึกษา', 'จบแล้วไปทำงานอะไรได้บ้าง']
delw_23 = ['อาจารย์','ครู','ติดต่อ', 'เบอร์โทร', 'เบอร์', 'เว็บไซต์', 'เมลล์', 'อีเมลล์', 'และ', 'เว็บไซต์']
# n_class = 11
# history = np.load('./models/model_history_2_0.9375.npy', allow_pickle='TRUE').item()
word_vector_length = 300
max_sentence_length = 20
ner = ThaiNameTagger()
wvmodel = word_vector.get_model()
cat_token_0 = [pos_tag(word_tokenize(sent)) for sent in category_0]
cat_token_2 = [pos_tag(word_tokenize(sent)) for sent in category_2]
cat_token_3 = [pos_tag(word_tokenize(sent)) for sent in category_3]
cat_token_5 = [pos_tag(word_tokenize(sent)) for sent in category_5]
cat_token_7 = [pos_tag(word_tokenize(sent)) for sent in category_5[:3]]

file = open('corpus/common_words.csv', 'r', encoding='utf-8')
data = list(csv.reader(file))
common_words = [d[0] for d in data]
trans_words = [d[1] for d in data]
str_row_instructor = np.array([len(d) for d in data]).argmax()


def words2vec(question, max_sentence_length, word_vector_length, wvmodel):
    pass_count = 0
    sents = [question]
    words = [[w for w in word_tokenize(s) if w != ' '] for s in sents]
    word_vectors = np.zeros((len(words), max_sentence_length, word_vector_length))
    sample_count = 0
    for s in words:
        word_count = 0
        for w in s[::-1]:
            try:
                word_vectors[sample_count, max_sentence_length - word_count - 1, :] = wvmodel[w]  # wvmodel_pythainlp
                word_count = word_count + 1
            except:
                pass_count += 1
                pass
        sample_count = sample_count + 1
    return word_vectors, words, pass_count


# ___________________________________________ API ___________________________________________
@app.route('/get', methods=['GET'])
def read():
    # data = frequencyQ_ref.document(str(i)).get()
    qs = []
    f_count = []
    for i in range(len(f_questions)):
        if i < len(f_questions):
            data = frequencyQ_ref.document(str(i)).get()
            f_count.append(data.to_dict()['count'])
    for _ in range(3):
        x = np.array(f_count).argmax()
        qs.append(f_questions[x])
        f_count[x] = -1
    return jsonify({'questions': qs})

@app.route('/save', methods=['POST'])
def save():
    try:
        QnA_ref.document().set(request.json)
        if int(request.json['tag']) < len(f_questions):
            data = frequencyQ_ref.document(request.json['tag']).get()
            count = data.to_dict()['count'] + 1
            frequencyQ_ref.document(request.json['tag']).update({'count':count}.json)
        return jsonify({"success": True}), 200
    except Exception as e:
        return f"An Error Occured: {e}"

@app.route('/', methods=['POST'])
def ans():
    """
        read() : Fetches documents from Firestore collection as JSON
        todo : Return document that matches query ID
        all_todos : Return all documents
    """
    # [1:'ประวัติ', 2:'ที่อยู่', 3:'ติดต่อ', 4:'บุคลากร', 5:'ข่าว', 6:'หลักสูตร',
    # 6:'เกี่ยวกับการเรียน', 7:'เข้าศึกษา', 8:'อาชีพ', 9:'ไม่เกี่ยวข้อง', 10:'ทักทาย']

    if (not request):
        return 'no request.'
    question = request.json['Q'].replace(' ', '')
    question_c = ''
    words_set = {''}
    for word in word_tokenize(question):
        if word.lower() in common_words[:str_row_instructor]:
            word = trans_words[common_words.index(word.lower())]
        if word not in words_set:
            question_c += word
            words_set.add(word)
    # print(question_c)
    # normalize(question_c)
    word_vectors, words, pass_count = words2vec(question_c, max_sentence_length, word_vector_length, wvmodel)
    n = model.predict(word_vectors).argmax(axis=1)[0]
    if n != 2 and n != 3 and 2*pass_count >= len(words):
        n = 9
    print('class :', n, words)
    if n == 0:
        score = [0] * len(category_0)
        history = chatbot_ref.document('history').get()
        for w, t in pos_tag(word_tokenize(question_c)):
            if t == 'CMTR':
                score[1] += 1
                score[2] += 1
            if w != '' and t == 'NCMN':
                for i, cat in enumerate(cat_token_0):
                    for c, _ in cat:
                        try:
                            score[i] += wvmodel.similarity(w, c)
                        except:
                            pass
        c = np.array(score).argmax()
        # [0:'ประวัติ', 1:'อายุ cpe', 2:'อายุ isne', 3:'รุ่น cpe', 4:'รุ่น isne']
        if c == 0 or c == 1:
            return jsonify({
                'tag': str(n),
                'A1': f'ภาควิชาคอมพิวเตอร์ ได้ถูกก่อตั้งขึ้นเมื่อ พ.ศ. {int(history.to_dict()["cpe"])}',
                'A2': f'ปัจจุบันมีอายุ {int(date.today().year)-int(history.to_dict()["cpe"])+543} ปี'
            }), 200
        elif c == 1:
            return jsonify({
                'tag': str(n),
                'A1': f'ภาควิชาคอมพิวเตอร์ ปัจจุบันมีอายุ {int(date.today().year) - int(history.to_dict()["cpe"]) + 543} ปี',
                'A2': f'ถูกก่อตั้งขึ้นเมื่อ พ.ศ. {int(history.to_dict()["cpe"])}'
            }), 200
        elif c == 2:
            return jsonify({
                'tag': str(n),
                'A1': f'ภาควิชาสารสนเทศ สื่อสารและเครือข่าย ปัจจุบันมีอายุ {int(date.today().year) - int(history.to_dict()["isne"]) + 543} ปี',
                'A2': f'ถูกก่อตั้งขึ้นเมื่อ พ.ศ. {int(history.to_dict()["isne"])}'
            }), 200
        elif c == 3:
            return jsonify({
                'tag': str(n),
                'A1': f'CPE รุ่นที่ {int(date.today().year) - int(history.to_dict()["cpe"]) + 544}',
            }), 200
        else:
            return jsonify({
                'tag': str(n),
                'A1': f'ISNE รุ่นที่ {int(date.today().year) - int(history.to_dict()["isne"]) + 544}'
            }), 200
    elif n == 1:
        address = chatbot_ref.document('address').get()
        return jsonify({
            'tag': str(n),
            'A1': address.to_dict()['name'],
            'A2': address.to_dict()['address']
        }), 200
    elif n == 2:
        contect = chatbot_ref.document('contect').get()
        score = [0] * len(category_2)
        for w, t in pos_tag(word_tokenize(question_c)):
            if w != '' and t == 'NCMN':
                for i, cat in enumerate(cat_token_2):
                    for c, _ in cat:
                        try:
                            score[i] += wvmodel.similarity(w, c)
                        except:
                            pass
        c = np.array(score).argmax()
        if c == 0:
            return jsonify({
                'tag': str(n),
                'A1': f'สามารถติดต่อภาควิชาได้ดังนี้',
                'A2': f'อีเมลล์: {contect.to_dict()["email"]}\n'
                      f'โทรศัพท์: {contect.to_dict()["tel"]}\n'
                      f'แฟกซ์: {contect.to_dict()["fax"]}\n'
                      f'เว็บไซต์: {contect.to_dict()["website"]}\n'
                      f'facebook: {contect.to_dict()["facebook"]}'
            }), 200
        else:
            for w in delw_23:
                question_c = question_c.replace(w, '')
            question_x = word_tokenize(question_c)
            try:
                if question_x == []:
                     raise Exception('')
                name = data[np.array([[sum([fuzz.ratio(w, word_tokenize(d[1])) for w in question_x]) for d in data[str_row_instructor:]],
                                      [sum([fuzz.ratio(w, word_tokenize(d[0])) for w in question_x]) for d in data[str_row_instructor:]]])
                                      .max(axis=0).argmax() + str_row_instructor]

                return jsonify({
                    'tag': str(n),
                    'A1': {'key': f'อาจารย์ {name[1]}', 'value': f'{contect.to_dict()["instructor"]}{name[0].split()[0]}'}
                }), 200
            except:
                return jsonify({
                    'tag': str(n),
                    'A1': {'key': f'ข้อมูลการติดต่ออาจารย์', 'value': f'{contect.to_dict()["instructor_all"]}'}
                }), 200
    elif n == 3:
        contect = chatbot_ref.document('contect').get()
        score = [0] * len(category_3)
        for w, t in pos_tag(word_tokenize(question_c)):
            if t == 'CNIT':
                score[2] += 1
            if t == 'DIBQ':
                score[2] += 1
            if w != '' and t == 'NCMN':
                for i, cat in enumerate(cat_token_3):
                    for c, _ in cat:
                        try:
                            score[i] += wvmodel.similarity(w, c)
                        except:
                            pass
        # print(score)
        c = np.array(score).argmax()
        if c < 2:
            try:
                name = data[np.array([fuzz.ratio(category_3[c], d[2]) for d in data[str_row_instructor:]]).argmax() + str_row_instructor]
                return jsonify({
                    'tag': str(n),
                    'A1': {'key': f'{name[2]} {name[1]}', 'value': f'{contect.to_dict()["instructor"]}{name[0].split()[0]}'}
                }), 200
            except:
                return jsonify({
                    'tag': str(n),
                    'A1': {'key': f'ข้อมูลการติดต่ออาจารย์', 'value': f'{contect.to_dict()["instructor_all"]}'}
                }), 200
        elif c == 2:
            return jsonify({
                'tag': str(n),
                'A1': f'มีอาจารย์ทั้งหมด {len(common_words[str_row_instructor:])} คน',
                'A2': {'key': f'ข้อมูลการติดต่ออาจารย์', 'value': f'{contect.to_dict()["instructor_all"]}'}
            }), 200
        else:
            for w in delw_23:
                question_c = question_c.replace(w, '')
            question_x = word_tokenize(question_c)
            try:
                if question_x == []:
                     raise Exception('')
                name = data[np.array([[sum([fuzz.ratio(w, d[0]) for w in question_x]) for d in data[str_row_instructor:]],
                                      [sum([fuzz.ratio(w, d[1]) for w in question_x]) for d in data[str_row_instructor:]],
                                      [sum([fuzz.ratio(w, d[2]) for w in question_x]) for d in data[str_row_instructor:]]]
                                     ).max(axis=0).argmax() + str_row_instructor]

                return jsonify({
                    'tag': str(n),
                    'A1': {'key': f'{name[1]}', 'value': f'{contect.to_dict()["instructor"]}{name[0].split()[0]}'}
                }), 200
            except:
                return jsonify({
                    'tag': str(n),
                    'A1': {'key': f'ข้อมูลการติดต่ออาจารย์', 'value': f'{contect.to_dict()["instructor_all"]}'}
                }), 200

    elif n == 4:
        news = chatbot_ref.document('news').get()
        return jsonify({
            'tag': str(n),
            'A1': {'key': f'ติดตามได้ที่ข่าวประชาสัมพันธ์', 'value': f'{news.to_dict()["url"]}'}
        }), 200
    elif n == 5:
        score = [0] * len(category_5)
        for w, t in pos_tag(word_tokenize(question_c)):
            if w != '' and t == 'NCMN':
                for i, cat in enumerate(cat_token_5):
                    for c, _ in cat:
                        try:
                            score[i] += wvmodel.similarity(w, c)
                        except:
                            pass
        print(score)
        c = np.array(score).argmax()
        if c == 0:
            cpe_curriculum = chatbot_ref.document('cpe-curriculum').get()
            isne_curriculum = chatbot_ref.document('isne-curriculum').get()
            return jsonify({
                'tag': str(n),
                'A1': f'{cpe_curriculum.to_dict()["name"]} ',
                'A2': f'{cpe_curriculum.to_dict()["duration"]} '
                      f'ค่าเทอม {cpe_curriculum.to_dict()["tuition"]} ',
                'A3': {'key': f'หลักสูตรวิศวกรรมคอมพิวเตอร์', 'value': f'{cpe_curriculum.to_dict()["url"]}'},

                'A4': f'{isne_curriculum.to_dict()["name"]} ',
                'A5':     f'{isne_curriculum.to_dict()["duration"]} '
                      f'ค่าเทอม {isne_curriculum.to_dict()["tuition"]} ',
                'A6': {'key': f'หลักสูตรวิศวกรรมระบบสารสนเทศและเครือข่าย',
                       'value': f'{isne_curriculum.to_dict()["url"]}'}
            }), 200
        elif c == 1:
            mcpe_curriculum = chatbot_ref.document('mcpe-curriculum').get()
            return jsonify({
                'tag': str(n),
                'A1': f'{mcpe_curriculum.to_dict()["name"]}',
                'A2': f'{mcpe_curriculum.to_dict()["duration"]} '
                      f'ค่าเทอม {mcpe_curriculum.to_dict()["tuition"]}',
                'A3': {'key': f'หลักสูตรปริญญาโทของภาควิชาคอมพิวเตอร์', 'value': f'{mcpe_curriculum.to_dict()["url"]}'}
            }), 200
        elif c == 2:
            phd_curriculum = chatbot_ref.document('phd-curriculum').get()
            return jsonify({
                'tag': str(n),
                'A1': f'{phd_curriculum.to_dict()["name"]} ',
                'A2': f'{phd_curriculum.to_dict()["duration"]} '
                      f'ค่าเทอม {phd_curriculum.to_dict()["tuition"]} ',
                'A3': {'key': f'หลักสูตรปริญญาเอกของภาควิชาคอมพิวเตอร์', 'value': f'{phd_curriculum.to_dict()["url"]}'}
            }), 200
        elif c == 3:
            cpe_curriculum = chatbot_ref.document('cpe-curriculum').get()
            isne_curriculum = chatbot_ref.document('isne-curriculum').get()
            return jsonify({
                'tag': str(n),
                'A1': f'{cpe_curriculum.to_dict()["name"]} {cpe_curriculum.to_dict()["duration"]}',
                'A2': {'key': f'หลักสูตรของภาควิชาคอมพิวเตอร์', 'value': f'{cpe_curriculum.to_dict()["url"]}'},
                'A3': f'{isne_curriculum.to_dict()["name"]} {isne_curriculum.to_dict()["duration"]}',
                'A4': {'key': f'หลักสูตรของภาควิชาสารสนเทศ สื่อสารและเครือข่าย',
                       'value': f'{isne_curriculum.to_dict()["url"]}'}
            }), 200
        else:
            cpe_curriculum = chatbot_ref.document('cpe-curriculum').get()
            isne_curriculum = chatbot_ref.document('isne-curriculum').get()
            return jsonify({
                'tag': str(n),
                'A1': f'{cpe_curriculum.to_dict()["name"]} ',
                'A2': f'ค่าเทอม {cpe_curriculum.to_dict()["tuition"]}',
                'A3': {'key': f'หลักสูตรของภาควิชาคอมพิวเตอร์', 'value': f'{cpe_curriculum.to_dict()["url"]}'},
                'A4': f'{isne_curriculum.to_dict()["name"]} ',
                'A5': f'ค่าเทอม {isne_curriculum.to_dict()["tuition"]}',
                'A6': {'key': f'หลักสูตรของภาควิชาสารสนเทศ สื่อสารและเครือข่าย',
                       'value': f'{isne_curriculum.to_dict()["url"]}'}
            }), 200
    elif n == 6:
        learning = chatbot_ref.document('learning').get()
        return jsonify({
            'tag': str(n),
            'A1': f'{learning.to_dict()["quote"]}',
            'A2': f'{learning.to_dict()["details"]}',
            'A3': {'key': f'ข้อมูลเพิ่มเติม', 'value': f'{learning.to_dict()["url"]}'}
        }), 200
    elif n == 7:
        application = chatbot_ref.document('application').get()
        score = [0] * len(cat_token_7)
        for w, t in pos_tag(word_tokenize(question_c)):
            if w != '' and t == 'NCMN':
                for i, cat in enumerate(cat_token_7):
                    for c, _ in cat:
                        try:
                            score[i] += wvmodel.similarity(w, c)
                        except:
                            pass
        c = np.array(score).argmax()
        if c == 0 or np.array(score).max() < 1.5:
            return jsonify({
                'tag': str(n),
                'A1': f'{application.to_dict()["cpe"]}',
                'A2': {'key': f'TCAS(ภาควิชาคอมพิวเตอร์)', 'value': f'{application.to_dict()["url-cpe"]}'},
                'A3': f'{application.to_dict()["isne"]}',
                'A4': {'key': f'TCAS(ภาควิชาสารสนเทศ สื่อสารและเครือข่าย)',
                       'value': f'{application.to_dict()["url-isne"]}'},
                'A5': {'key': f'IPAS(ภาควิชาสารสนเทศ สื่อสารและเครือข่าย)',
                       'value': f'{application.to_dict()["url-isne-ipas"]}'}
            }), 200
        elif c == 1:
            return jsonify({
                'tag': str(n),
                'A1': {'key': f'การรับนักศึกษาปริญญาโท', 'value': f'{application.to_dict()["url-mcpe"]}'},
            }), 200
        else:
            return jsonify({
                'tag': str(n),
                'A1': {'key': f'การรับนักศึกษาปริญญาเอก', 'value': f'{application.to_dict()["url-phd"]}'},
            }), 200
    elif n == 8:
        job = chatbot_ref.document('job').get()
        return jsonify({
            'A1': job.to_dict()["details"],
            'A2': {'key': f'เพิ่มเติม', 'value': f'{job.to_dict()["url"]}'}
        }), 200
    elif n == 9:
        return jsonify({'A1': 'ฉันตอบไม่ได้'}), 200
    else:
        try:
            greeting = ['สวัสดี', 'สวัสดีจ้า', 'สวัสดีครับ']
            return jsonify({'tag': str(n), 'A1': greeting[random.randint(0, len(greeting) - 1)], 'A2': ' มีอะไรสอบถามไหมครับ'}), 200
        except Exception as e:
            return f"An Error Occured: {e}"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT')))