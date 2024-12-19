#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:53:20 2024

class modules for asset calculator

@author: jmr
"""

import pandas as pd
import numpy as np
import yfinance as yf
import time

# Utility functions

def rd_ticker_file(f_name):
    # Read tickers and returned a (cleaned) ticker list

    data_file=open(f_name,"r")
    data_lst=data_file.readlines()
    data_file.close()
    
    data_lst=[tk.strip('\n\t') for tk in data_lst]

    return data_lst


def lst2str(lst,separator=" "):
    # Convert list into one string contaning all elements in list

    out_str=""
    
    for tk in lst: 
        out_str+=tk+separator

    return out_str

def save_ticker_lst(fn,ticker_lst):
    # Saves ticker list to file 
        
    pr_str=lst2str(ticker_lst,'\n')

    output_ticker_file=open(fn,"w")

    output_ticker_file.write(pr_str)

    output_ticker_file.close()
    
def get_sector_lst(ticker_lst,batch=500,delay_sec=5):
    """
    Collects sector list and returns dictionary with sector and list with sector codes

    Parameters
    ----------
    ticker_lst : list
        Contains tickers to be assigned sector

    Returns
    -------
    tk_sectorid_dict : dictiornary
        Dictionary with ticker and related sectorid
    sector_id_dict : dictornary
        List with all sectors and sector id
    sector_lst : list
        list of sectors i.e. sector_lst[id]=sector

    """
    
    tk_sectorid_pd=pd.DataFrame(columns=['Ticker','Sectorid'])
    p_tk_sectorid_pd=0
    sector_id_dict={}
    sector_lst=[]
    no_calls=0
    valid_tk_lst=[]
    err_tk_lst=[]

    
    p_sector_id=0
    
    for tk in ticker_lst:
        try: 
            tk_obj=yf.Ticker(tk)
            sector=tk_obj.info['sector']
            if sector_lst.count(sector)==0:
                sector_lst.append(sector)
                tk_sectorid_pd.loc[p_tk_sectorid_pd]=[tk, p_sector_id]
                sector_id_dict[sector]=p_sector_id
                p_sector_id+=1
            else:
                sector_id=sector_id_dict[sector]
                tk_sectorid_pd.loc[p_tk_sectorid_pd]=[tk, sector_id]
            p_tk_sectorid_pd+=1
            valid_tk_lst.append(tk)
            if no_calls>batch: 
                no_calls=0
                time.sleep(delay_sec)
            else:
                no_calls+=1
        except:
            err_tk_lst.append(tk)    
            
    return tk_sectorid_pd,sector_lst,valid_tk_lst,err_tk_lst
        
    
def int2binvec(xx,vec_lgth=None):
    """
    Translate number xx into binary vector of length vec_lgth

    Parameters
    ----------
    xx : int
        Number to be translated
    vec_lgth : int>0, optional
        Length of binary vector. If not provided the minimum length is calculated. The default is None.

    Returns
    -------
    bin_vec : numpy(vec_lgth,dtype=float)
        Binary vector with result i.e. xx=<bin_vec;[1 2 4 8 16 ...2^vec_lgth]

    """
    
    # Determine lentgh of vector if not provided
    
    max_index=int(np.log(xx+1)/np.log(2))+1
    
    if vec_lgth==None: 
        vec_lgth=max_index
        
    # assert (max_index<=vec_lgth), f'int2binvec: max_index={max_index}>vec_lgth={vec_lgth} '
        
    # Determine binary vector
    
    res=xx
    bin_vec=np.zeros(vec_lgth,dtype=int)
    

    nn=max_index
    bb=2**max_index
    
    for ii in range(max_index+1):
        if res>=bb: 
            bin_vec[nn]=1
            res-=bb
        bb/=2
        nn-=1
        
    # assert (res==0), f'int2binvec: res={res}>0'
        
    return bin_vec
           
           
    

class asset_data(): 
    """
    This class contains functions to collect asset price data and transform these into return data.
    The following functins are availble:
        - load_price_data_from_file: file_name->table with prices (panda table)
        - load_price_data_from_yfinance: ticker_file -> table with prices (panda table)
        - save_price_data_to_file: table wih prices (panda), file_name->csv file
        - _transform_price2return_data: 
        - mean: rolling_window (optional), mememory_loss_factor (optional)->table (panda)
    """
    
    def __init__(self):
        """
        Iitialisation of class

        Returns
        -------
        None.

        """
        
    def sort_ticker_lst(self,ticker_src_lst):
        """
        This function checks the source ticker list for errors in the ticker labels i.e. 
        the ticker cannot be recognised by yfinance

        Parameters
        ----------
        ticker_src_lst : list of text strings
            Contains source ticker list

        Returns
        -------
        ticker_valid_lst : list of text strings
            Contains all valid tickers
        ticker_err_lst : list of text stings
            Contains all non-valid tickers
        

        """
        
        # Initialise
        
        ticker_valid_lst=[]
        ticker_err_lst=[]
        
        # check if all tickers are valid
        
        for tk in ticker_src_lst: 
            try: 
                if len(yf.Ticker(tk).history(period='7d',interval='1d'))==0: 
                    ticker_err_lst.append(tk)
                else: 
                    ticker_valid_lst.append(tk)
            except: 
                ticker_err_lst.append(tk)
                
        # return ticker lists

        return ticker_valid_lst,ticker_err_lst     

    def _split_lst(self,src_lst,batch_size=500):
        """
        
        Split list into sub list of size <=batch_size

        Parameters
        ----------
        src_lst : TYPE
            DESCRIPTION.
        batch_size : TYPE, optional
            DESCRIPTION. The default is 500.

        Returns
        -------
        None.

        """
        
        aggr_lst=[]
        lgth_src_lst=len(src_lst)
        
        for ii in range(0,lgth_src_lst,batch_size):
            if ii+batch_size<lgth_src_lst: 
                aggr_lst.append(src_lst[ii:ii+batch_size])
            else:
                aggr_lst.append(src_lst[ii:])
            
        return aggr_lst
            
        
   
    def load_ticker_data_from_yfinance(self,ticker_lst,from_dt,to_dt,batch_size=500,\
                                       delay_sec=5,select_col=None,data_label=None):
        """
        Collect price data from yahoo applying yfinance

        Parameters
        ----------
        ticker_lst : list of text strings
            List of ticker (assumes all tickers are valid)
        from_dt : date (yyyy-mm-dd)
            From date
        to_dt : date (yyyy-mm-dd)
            To date
        data_label : text string, optional
            Label that defines columns to apply as data. The default is None.

        Returns
        -------
        
        price_data_pd : panda table
            Panda table with asset price data

        """
        
        # Download data
        
        price_data_pd=pd.DataFrame()
        valid_ticker_lst=[]
        err_ticker_lst=[]
            
        tk_sets=self._split_lst(ticker_lst,batch_size)
            
        for ii in range(0,len(tk_sets)):
            
            try: 
                
                data_e_pd=yf.download(tk_sets[ii],start=from_dt,end=to_dt,group_by='ticker',\
                                  keepna=True,progress=False)
                    
                if not(data_e_pd.empty): 
                    if select_col!=None: 
                        if batch_size>1: 
                            data_e_pd=self.select_columns_in_data_pd(data_e_pd,select_label=select_col)
                        else:
                            data_e_pd=self.select_columns_in_data_pd(data_e_pd,select_label=select_col,ticker=tk_sets[ii])
                    price_data_pd=pd.concat([price_data_pd,data_e_pd],axis=1,sort=True)
                    valid_ticker_lst+=tk_sets[ii]
                else: 
                    err_ticker_lst+=tk_sets[ii]
                
                time.sleep(delay_sec)
                
            except: 
                err_ticker_lst+=tk_sets[ii]
                
                
            
            
        # Create table with relevant data
        
        if data_label!=None: 
            mindex=[(tk,data_label) for tk in ticker_lst]
            price_data_pd=price_data_pd[mindex]
        else:
            price_data_pd=price_data_pd
                

            
        # Interpolate data where Nan's
        
        price_data_pd=price_data_pd.interpolate(axis=0,limit_direction='both')
        

        # Return data
        
        return price_data_pd, valid_ticker_lst,err_ticker_lst
    
        
        
    def select_columns_in_data_pd(self,data_pd,select_label=None,ticker=None):
        """
        Drop columns in the table from column id from_column that is not an 
        id=from_id+p*step_column, p in )0,1,2,...
        Parameters
        ----------
        data_pd : panda table
            Panda table with data
        from_column : int
            column id
        step_column : int
            column id step
        start_column : int
            first column that may be droppped

        Returns
        -------
        new_data_pd : panda table 
            Reduced panda table

        """
        

        if ticker==None: 
            
            res_pd=pd.DataFrame()
            col_names=[]
        
            for tk,label in data_pd.columns: 
                if label==select_label: 
                    res_pd=pd.concat([res_pd,data_pd[tk,label]],axis=1)
                    col_names.append(tk)
                
            res_pd.columns=col_names
            
        else:
            
            res_pd=pd.DataFrame()
            res_pd=pd.concat([res_pd,data_pd[select_label]],axis=1)
            res_pd.columns=ticker
        
                
        return res_pd
        
        
    def transform_price2return_data(self,price_data_pd,ln_conv=True,fx_conv_matr=None):
        """
        Transform price data in panda table to numpy array by the following formula: 
            x[ii,jj]=log(pd[ii,jj])-log(pd[ii-1,jj])

        Parameters
        ----------
        price_data_pd : panda table
            Contains price data (assumption is that all prices>0)
        fx_conv_matr : numpy array (), optional    
            Matrix that are multiplied on transformed price data

        Returns
        -------
        price_data_conv : numpy array
            Converted price data

        """
        
        # transform panda data
        
        if ln_conv:
            
            try: 
        
                c_price_data_pd=price_data_pd.apply(np.log)
                
            except: 
                
                print('transform_price2return_data : Something went wrong in ln convertion')
                
        else: 
            
            c_price_data_pd=price_data_pd
        
        # transform to numpy array
        
        c_price_data=c_price_data_pd.to_numpy(dtype=float)
        
        # Take 1 lag difference in time dimension
        
        sh=c_price_data.shape
        sh[0]-=1
        di_mat=np.zeros(sh,dtype=float)
        
        for ii in range(sh[0]):
            di_mat[ii,ii]=-1
            di_mat[ii,ii+1]=1
            
        d_c_price_data=di_mat@c_price_data
        
        # apply fx_conv_matr if applicable
        
        if fx_conv_matr!=None: 
            d_c_price_data=d_c_price_data@fx_conv_matr
            
        # return converted data
        
        return d_c_price_data
    

        
    def load_price_data_from_file(self,data_file_name):
        """
        Load price data from csv file and reurns as panda dataframe

        Parameters
        ----------
        data_file_name : string
            Datafile name: data_file_name+".csv"

        Returns
        -------
        price_data_pd : panda dataframe
        Panda dataframe with asset price data. If None something went wron
        
        """
        
        try: 
            
            price_data_pd=pd.read_csv(data_file_name+'.csv',low_memory=False)
            
        except: 
            
            price_data_pd=None
            
        return price_data_pd     
        
    def save_price_data_to_file(self,data_file_name,price_data_pd):
        """
        Save price data to csv file. If succesfull +1 is returned otherwise -1

        Parameters
        ----------
        data_file_name : string
            Data file name will be: data_file_name+'.csv'
        price_data_pd : Panda dataframe
            Contains price data

        Returns
        -------
        ret_val : int
        If succesfull +1 is returned otherwise -1

        """
        
        try: 
            price_data_pd.to_csv(data_file_name+'.csv')
            ret_val=1
            
        except: 
            ret_val=-1
            
            
        return ret_val    
    
    
    def save_return_data_to_file(self,data_file_name,return_data_array,header=''):
        """
        Save return data in numpy array to csv file

        Parameters
        ----------
        data_file_name : string
            File name with data is: data_file_name+'.csv'
        return_data_array : numpy array with return data
            Array containing transformed price data to return data in the format of a numpy array
            

        Returns
        -------
        ret_val : int
        Succesful storing of data -1: something went wrong

        """
        
        try: 
            
            np.savetxt(data_file_name+'.csv',return_data_array,header=header)
            ret_val=1
         
        except: 
            ret_val=-1
            
            
        return ret_val


    def load_return_data_from_file(self,data_file_name):
        """
        Load return data in csv file into numnpy array

        Parameters
        ----------
        data_file_name : string
            Data file name is: data_file_name+'.csv'

        Returns
        -------
        return_data_array: numpy array
        If succesful return data is stored in numpy array: return_data_array otherwise return_data_array=None

        """
            
        try: 
            
            return_data_array=np.loadtxt(data_file_name+'.csv',dtype=float)
            
        except: 
            
            return_data_array=None
            
            
        return return_data_array



class asset_model():
    """
      This class contains functions that calculates parameters for a generalised linear model that
      predicts dtd return for assets given dtd return for factors. The generalised model can be 
      static or adaptive.
      In case of adaptive model statistics can be calculated on the estimated parameters.
        
     """
      
    def __init__(self):
        """ 
        initialises the model
        
        """
        
        pass
      
    def calc_rolling_stat(self,data_array,data_window=1):
        """
        Calculates rolling average and variance of data in data_array across rows (axis=0)

        Parameters
        ----------
        data_array : numpy array
            2 dimensional array wihth data
            
        data_window : int
            Number of observations in each calculation of average


        Returns
        -------
        avg_data_array : numpy array
           Contains rolling average
        var_data_array : numpy array
           Contains rolling variace

        """
        
        # initialise
        
        
        sh=(data_array.shape[0]-data_window+1,data_array.shape[1])
        
        avg_data_array=np.zeros(sh,dtype=float)
        var_data_array=np.zeros(sh,dtype=float)
        
        tr_mat=np.zeros((data_array.shape[0],data_array.shape[0]),dtype=float)
        
        for ii in range(data_window,data_array.shape[0]):
            tr_mat[ii,ii-data_window]=1
        
        
        data2_array=data_array**2 # squared values of data
        
        # Calculates accumulated sum 
        
        cs=np.cumsum(data_array,axis=0)
        cs2=np.cumsum(data2_array,axis=0)
        
        # Calculates rolling values
        
        mm=(cs-tr_mat@cs)/data_window
        mm2=(cs2-tr_mat@cs2)/data_window
        
        avg_data_array[:,:]=mm[data_window-1:,:]
        var_data_array[:,:]=(mm2-mm**2)[data_window-1:,:]
        
        # return result
        
        return avg_data_array,var_data_array
    
    
        
    def _solve_linear_adaptive_model(self,LHS_mat,RHS_mat,init_n_row=None,depr_param=0.0):
        """
        Solve a generalised linear model adaptively. First init_n_row solutions will be the same.

        Parameters
        ----------
        LHS_mat : Numpy array((T,n),float)
            Represent LHS
        RHS_mat : Numpy array((T,m),float)
            Represent RHS
        init_n_row : int, optional
            Number of rows applied for first solution>=n. The default is None i.e. n is applied
        depr_param : float (0,1(, optional
            Depreciation parameter for error term. The default is 0.0. i.e. equal weighting of error terms

        Returns
        -------
        x : Numpy array((T,n,m),float)
           Optimal parameters. Note the fist init_n_row is the same
        err_RHS: Numpy array((T,m),float)
           Error based on difference between RHS and predicted RHS based on estimated parameters 
           estimated just before row with RHS

        """
        
        #------ Initialisation
        
        # initialise data structures
        
        (T,n)=LHS_mat.shape
        (T1,m)=RHS_mat.shape
        
        # assert T!=T1, 'solve_linear_adaptive_model: Number of rows between LHS and RHS do not match!'
        # assert n>T, 'Number of rows in LHS is less than number of columns'
        
        x=np.zeros((T,n,m),dtype=float)
        P=np.zeros((T,n,n),dtype=float)
        err_RHS=np.zeros((T,m),dtype=float)
        theta=1.0-depr_param
        
        if init_n_row==None: 
            kk=n-1
        elif init_n_row<n: 
            kk=n-1
        else:
            kk=init_n_row-1
        
        
        # initialise solution
        
        P[kk]=LHS_mat[:(kk+1),:].transpose()@LHS_mat[:(kk+1),:]
        x[kk]=np.linalg.solve(P[kk],LHS_mat[:(kk+1),:].transpose()@RHS_mat[:(kk+1),:])
        
        for ii in range(kk):
            P[ii]=P[kk]
            x[ii]=x[kk]
            
        # ---- Estimate the adaptive linear model    
        
        for ll in range(kk+1,T):
            P[ll]=theta*P[ll-1]+np.expand_dims(LHS_mat[ll,:],axis=1)@np.expand_dims(LHS_mat[ll,:],axis=0)
            err_RHS[ll,:]=RHS_mat[ll,:]-LHS_mat[ll,:]@x[ll-1]
            x[ll]=x[ll-1]+np.linalg.solve(P[ll],np.expand_dims(LHS_mat[ll,:],axis=1)@np.expand_dims(err_RHS[ll,:],axis=0))
            
        # return solution
       
        return x,err_RHS
            
            
    def calibrate_adaptive_linear_model(self,return_data_mat,no_factor,depr_param=0.0,init_window=None,data_window=1):
        """
        Calibrate an adaptive generalised linear factor model 

        Parameters
        ----------
        return_data_mat : numpty array((T,n+m),float)
            Matrix including dtd return data for factors and assets. First (n-1) columns contains 
            factor returns and the remaning m columns contains asset returns
        no_factor : int (=n)
            Number of factors
        depr_param : float, optional
            Depreciation parameter in adaptive model. The default is 0.0.
        init_window : int>0, optional
            Number of observations for the initial calibration in the adaptive model. The default is None.
            In case of None then the initial number of data for initial calibration is set to 2*(number of factors)
        data_window : int, optional
            No of observations in rolling mean and variance of parameters

        Returns
        -------
        param : numpy array ((T,n+1,m),float)
         Column j contains parameters for asset j (=0..(m-1)), where row i=0 is alpha (constant) and i=1..n 
         contain beta for factor (i-1) 
        pred_err : Numpy array((T,m),float)
           Error based on difference between asset return and predicted asset return based on estimated parameters 
           estimated just before row with RHS
        stat_param : list (m elements)
           List containing pairs (mean,var) for each asset (0..(m-1)). mean and var are numpy arrays
           with structure ((T,n+1),float)
        var_err : numpy array (n+1,float)   
           Vector with variance of error term i.e. non-systematic risk

        """
        
        # initialise
        
        last_factor_index=no_factor-1
        
        no_asset=return_data_mat.shape[1]-no_factor
        
        no_data=return_data_mat.shape[0]
        
        return_factor_mat=return_data_mat[:,:last_factor_index+1]
        return_asset_mat=return_data_mat[:,last_factor_index+1:]
        
        mreturn_factor_mat=np.insert(return_factor_mat,0,1,axis=1)
        
        if init_window==None: 
            init_window=2*no_factor
        
        
        # calibrate adaptive model
        
        param,pred_err=self._solve_linear_adaptive_model(mreturn_factor_mat,return_asset_mat,\
                                                         init_window,depr_param)
            
        # Calculate mean and variance of parameters and prediction error
        
        stat_param=[self.calc_rolling_stat(param[:,:,asset_id],data_window) for asset_id in range(no_asset)]
        
        var_err=np.sum(pred_err**2,axis=0)/(no_data-(no_factor+1))    
            
        # return results
       
        return param,pred_err,stat_param,var_err
    
class portfolio_model():
      """
        This class contains functions to perform portfolio construction.
        The functions are: 
            
      """

      def __init__(self):
          """
        

          Returns
          -------
          None.

          """
        
          pass
      
        
    
    
    
   
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        