from ArticulatoryTextFrontend import ArticulatoryTextFrontend
from ArticulatoryTextFrontend import get_language_id

if __name__ == '__main__':
    # demonstrating the language ID lookup
    print(get_language_id("eng"))
    print(get_language_id("deu"))
    print(get_language_id("fra"))

    # demonstrating the conversion from graphemes to features
    print("\n\nEnglish Test")
    tf = ArticulatoryTextFrontend(language="eng")
    features = tf.string_to_tensor("This is a complex sentence, it even has a pause!", view=True)

    print("\n\nChinese Test")
    tf = ArticulatoryTextFrontend(language="cmn")
    features = tf.string_to_tensor("这是一个复杂的句子，它甚至包含一个停顿。", view=True)
    features = tf.string_to_tensor("李绅 《悯农》 锄禾日当午， 汗滴禾下土。 谁知盘中餐， 粒粒皆辛苦。", view=True)
    features = tf.string_to_tensor("巴 拔 把 爸 吧", view=True)

    print("\n\nVietnamese Test")
    tf = ArticulatoryTextFrontend(language="vie")
    features = tf.string_to_tensor("Xin chào thế giới, quả là một ngày tốt lành để học nói tiếng Việt!", view=True)
    features = tf.string_to_tensor("ba bà bá bạ bả bã", view=True)

    print("\n\nJapanese Test")
    tf = ArticulatoryTextFrontend(language="jpn")
    features = tf.string_to_tensor("医師会がなくても、近隣の病院なら紹介してくれると思います。", view=True)

    print("\n\nZero-Shot Test")
    tf = ArticulatoryTextFrontend(language="acr")
    features = tf.string_to_tensor("I don't know this language, but this is just a placeholder text anyway.", view=True)
