import inflect
import re


def df_groupby(df, cols):
    if not cols:
        return df
    else:
        gpobj = df.groupby(cols[0])
        group_dict = dict()
        for group in gpobj.groups.keys():
            new_df = gpobj.get_group(group)
            group_dict[group] = df_groupby(new_df, cols[1:])
        return group_dict


class MatchHelper(object):
    vowels = ('a', 'e', 'i', 'o', 'u')

    @staticmethod
    def get_words(string):
        return re.split('[ ,.?;:]+', string)

    @staticmethod
    def get_initials(string):
        words = MatchHelper.get_words(string)
        initials = list()
        for word in words:
            if word[0:2].lower() == 'ex':
                initials.append(word[1])
            elif re.match('[A-Za-z]', word[0]):
                initials.append(word[0])
        return initials

    @staticmethod
    def match_initials(s1, s2, casesensitive=False):
        if not casesensitive:
            s1 = s1.lower()
            s2 = s2.lower()
        return ''.join(MatchHelper.get_initials(s1)) == s2 \
               or ''.join(MatchHelper.get_initials(s2)) == s1

    @staticmethod
    def match_first_n(s1, s2, n=2, casesensitive=False):
        if not casesensitive:
            s1 = s1.lower()
            s2 = s2.lower()
        if len(s1) >= n and len(s2) >= n:
            return s1[0:n] == s2[0:n]
        elif len(s1) < n:
            return s1[0:] == s2[0:n]
        elif len(s2) < n:
            return s1[0:n] == s2[0:]
        return False

    @staticmethod
    def to_singular_noun(string, p=inflect.engine()):
        n_singluar = p.singular_noun(string)
        n_plural = p.plural_noun(string)
        plural = p.plural(string)

        if not n_singluar:
            return string
        elif n_singluar == plural:
            return n_singluar
        elif plural == n_plural:
            if n_singluar == plural[:-3]:
                return string
            else:
                return n_singluar
        else:
            return n_singluar

    @staticmethod
    def match_sgl_plrl(s1, s2, casesensitive=False, p=inflect.engine(), operator=lambda x, y: x == y):
        if not casesensitive:
            s1 = s1.lower()
            s2 = s2.lower()
        if operator(s1, s2):
            return True
        if len(s1) < 3:
            return False
        return operator(MatchHelper.to_singular_noun(s1, p), s2) or operator(p.plural_noun(s1), s2)


    @staticmethod
    def match_in_string(s_ref, s_sample, one=True, stemming=False, casesensitive=False, engine=inflect.engine()):
        if not casesensitive:
            s_ref = s_ref.lower()
            s_sample = s_sample.lower()

        if not one:
            return MatchHelper.match_sgl_plrl(s_sample, s_ref, True, engine, lambda x, y: x in y)
        else:
            wds_sample = MatchHelper.get_words(s_sample)
            wds_ref = MatchHelper.get_words(s_ref)

            found = False
            for ws in wds_sample:
                found = any(MatchHelper.match_sgl_plrl(ws, wr, True, engine) for wr in wds_ref) \
                    if stemming else ws in wds_ref
                if found:
                    return found
            return found

    # @staticmethod
    # def match_in_string(s_ref, s_sample, one=True, stemming=False, casesensitive=False, engine=inflect.engine()):
    #     if not casesensitive:
    #         s_ref = s_ref.lower()
    #         s_sample = s_sample.lower()
    #
    #     wds_sample = MatchHelper.get_words(s_sample)
    #     wds_ref = MatchHelper.get_words(s_ref)
    #     if not one:
    #         wds_sample = [' '.join(wds_sample)]
    #         wds_ref = ' '.join(wds_ref)
    #
    #     found = False
    #     for ws in wds_sample:
    #         found = ws in wds_ref
    #         if len(ws) < 3:
    #             continue
    #         if (not found) and stemming:
    #             found = (engine.plural(ws) in wds_ref)
    #             if not found:
    #                 sgl = engine.singular_noun(ws)
    #                 found = sgl in wds_ref if sgl else sgl
    #         if found:
    #             return found
    #     return found



# exchanges = ['asx', 'bloomberg', 'cme', 'cbot', 'nymex_comex', 'eurex', 'hkfe', 'ice', 'ose', 'sgx']
# report_fmtname = 'Web_ADV_Report_{}.xlsx'
#
# report_files = {e: report_fmtname.format(e.upper()) for e in exchanges}
#
# cmeg_prds_file = 'Product_Slate.xls'
# cmeg_adv_files = {CMEGMatcher.CME: report_files['cme'],
#                   CMEGMatcher.CBOT: report_files['cbot'],
#                   CMEGMatcher.NYMEX: report_files['nymex_comex']}
#
# cmeg = CMEGMatcher()
# dfs_adv = {k: pd.read_excel(v) for k, v in cmeg_adv_files.items()}
# df_prods = pd.read_excel(cmeg_prds_file)
# cmeg.run_pd_mtch(dfs_adv, df_prods, ('CME_Product_Index', 'CBOT_Product_Index'), clean=True)



# region TODO: test
# def __match_prods_by_commodity(self, df_prods, df_adv):
#     prod_dict = {str(row[self.GLOBEX]): row for _, row in df_prods.iterrows()}
#     prod_dict.update({str(row[self.CLEARING]): row for _, row in df_prods.iterrows()})
#     matched_rows = [{**row, **prod_dict[str(row[self.COMMODITY])]} for _, row in df_adv.iterrows()
#                     if str(row[self.COMMODITY]) in prod_dict]
#     cols = list(df_adv.columns) + list(df_prods.columns)
#     df = pd.DataFrame(matched_rows, columns=cols)
#     # unmatched = [(i, str(entry)) for i, entry in df_adv[self.COMMODITY].iteritems() if
#     #              str(entry) not in df_prods[self.GLOBEX].astype(str).unique()]
#     # unmatched = [(i, entry) for i, entry in unmatched if entry not in df_prods[self.CLEARING].astype(str).unique()]
#     # ytd = find_first_n(list(df_adv.columns), lambda x: self.PATTERN_ADV_YTD in x)
#     # indices = [i for i, _ in unmatched if df_adv.iloc[i][ytd] == 0]
#     # df_adv.drop(df_adv.index[indices], inplace=True)
#     # df_adv.reset_index(drop=0, inplace=True)
#     return df
# endregion

# region unused
# def __clean_prod_name(self, row):
    #     product = row[self.PRODUCT_NAME]
    #     der_type = row[self.CLEARED_AS]
    #     product = product.replace(der_type, '') if last_word(product) == der_type else product
    #     return product.rstrip()

# endregion