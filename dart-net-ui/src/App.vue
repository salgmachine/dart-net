<template>
  <div class="container">
    <div class="large-12 medium-12 small-12 cell">
      <label>
        File
        <input type="file" id="file" ref="file" v-on:change="handleFileUpload()">
      </label>
      <button v-on:click="submitFile()">Submit</button>
    </div>
    <div style="margin-top: 20px">
        <span v-show="!!response">{{response}}</span>
    </div>
  </div>
</template>

<script>
import axios from "axios";
export default {
  /*
      Defines the data used by the component
    */
  data() {
    return {
      file: "",
      response: undefined
    };
  },

  methods: {
    /*
        Submits the file to the server
      */
    submitFile: function() {
      var self = this;
      let formData = new FormData();

      formData.append("file", this.file);

      axios
        .post("/nets/darknet", formData, {
          headers: {
            "Content-Type": "multipart/form-data"
          }
        })
        .then(function(ok) {
          // eslint-disable-next-line no-console
          console.log("SUCCESS!! ", ok);
          self.response = ok.data;
        })
        .catch(function(e) {
          // eslint-disable-next-line no-console
          console.log("FAILURE!! ", e);
        });
    },

    /*
        Handles a change on the file upload
      */
    handleFileUpload() {
      this.file = this.$refs.file.files[0];
    }
  }
};
</script>

<style >
</style>